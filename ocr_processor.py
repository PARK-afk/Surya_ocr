import cv2
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
import re
import os
import time
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from surya.layout import LayoutPredictor
from paddleocr import PaddleOCR


class OCRProcessor:
	"""스마트 이미지 OCR 처리 클래스"""
	
	def __init__(self, max_height=2000, overlap_ratio=0.1, debug=False):
		"""
		Args:
			max_height: 이미지 분할 기준 최대 높이
			overlap_ratio: 분할 구간 간 중첩 비율
			debug: 디버그 모드
		"""
		self.max_height = max_height
		self.overlap_ratio = overlap_ratio
		self.debug = debug
		self.recognition_predictor = None
		self.detection_predictor = None
		self.paddle_ocr = None
	
	def _init_predictors(self):
		"""Surya 예측기 초기화 (지연 로딩)"""
		if self.recognition_predictor is None:
			from surya.foundation import FoundationPredictor
			foundation_predictor = FoundationPredictor()
			self.recognition_predictor = RecognitionPredictor(foundation_predictor)
			self.detection_predictor = DetectionPredictor()
	
	def _init_paddle_ocr(self, use_gpu=False):
		"""PaddleOCR 초기화 (지연 로딩)"""
		if self.paddle_ocr is None:
			self.paddle_ocr = PaddleOCR(use_textline_orientation=True, lang='korean')
	
	def smart_image_split(self, image_path):
		"""
		세로로 긴 이미지를 지능적으로 분할
		
		Returns:
			분할된 이미지들의 리스트와 각각의 Y 오프셋 정보
		"""
		img = Image.open(image_path)
		width, height = img.size
		
		if height <= self.max_height:
			if self.debug: 
				print(f"📏 분할 불필요: 높이 {height} <= {self.max_height}")
			return [(img, 0)]
		
		if self.debug: 
			print(f"📏 긴 이미지 감지: {width}x{height} -> 분할 시작")
		
		# 1단계: 빠른 레이아웃 분석으로 안전한 분할점 찾기
		safe_split_points = self._find_safe_split_points(img)
		
		# 2단계: 안전한 분할점을 기반으로 이미지 분할
		image_segments = []
		overlap_height = int(self.max_height * self.overlap_ratio)
		
		for i, (start_y, end_y) in enumerate(safe_split_points):
			# 중첩 영역 추가 (첫 번째와 마지막 제외)
			actual_start = max(0, start_y - (overlap_height if i > 0 else 0))
			actual_end = min(height, end_y + (overlap_height if i < len(safe_split_points) - 1 else 0))
			
			# 이미지 분할
			segment = img.crop((0, actual_start, width, actual_end))
			image_segments.append((segment, start_y))  # 원본에서의 Y 오프셋 저장
			
			if self.debug: 
				print(f"📸 분할 {i+1}: Y {actual_start}-{actual_end} (원본 기준: {start_y}-{end_y})")
		
		return image_segments

	def _find_safe_split_points(self, img):
		"""텍스트 영역을 피해서 안전한 분할점 찾기"""
		width, height = img.size
		
		# 간단한 방법: 가로 히스토그램으로 텍스트 밀도 분석
		gray_img = img.convert('L')
		img_array = np.array(gray_img)
		
		# 각 행의 픽셀 변화량 계산 (텍스트가 있으면 변화량이 큼)
		horizontal_projection = []
		for y in range(height):
			row = img_array[y, :]
			# 픽셀값 변화량 계산
			diff = np.sum(np.abs(np.diff(row.astype(int))))
			horizontal_projection.append(diff)
		
		# 변화량을 정규화
		max_diff = max(horizontal_projection) if horizontal_projection else 1
		normalized_projection = [x / max_diff for x in horizontal_projection]
		
		if self.debug:
			print(f"📊 가로 히스토그램 분석 완료")
		
		# 분할점 후보 찾기
		split_candidates = []
		current_y = 0
		
		while current_y < height:
			target_end = min(current_y + self.max_height, height)
			
			if target_end >= height:
				# 마지막 구간
				split_candidates.append((current_y, height))
				break
			
			# 목표 지점 주변에서 가장 안전한 분할점 찾기
			search_start = max(current_y + self.max_height - 200, current_y + self.max_height // 2)
			search_end = min(target_end + 200, height)
			
			best_split_y = self._find_best_split_in_range(
				normalized_projection, search_start, search_end
			)
			
			split_candidates.append((current_y, best_split_y))
			current_y = best_split_y
		
		if self.debug:
			print(f"📍 분할점 후보: {len(split_candidates)}개")
			for i, (start, end) in enumerate(split_candidates):
				print(f"  구간 {i+1}: {start} - {end} (높이: {end-start})")
		
		return split_candidates

	def _find_best_split_in_range(self, projection, start_y, end_y):
		"""주어진 범위에서 가장 안전한 분할점 찾기"""
		if start_y >= end_y or start_y >= len(projection):
			return end_y
		
		# 검색 범위 제한
		actual_start = max(0, start_y)
		actual_end = min(len(projection), end_y)
		
		# 범위 내에서 텍스트 밀도가 가장 낮은 지점 찾기
		min_density = float('inf')
		best_y = actual_end
		
		# 연속된 여러 행의 평균 밀도로 판단 (더 안정적)
		window_size = 5
		
		for y in range(actual_start, actual_end - window_size):
			# 윈도우 평균 계산
			window_avg = sum(projection[y:y+window_size]) / window_size
			
			if window_avg < min_density:
				min_density = window_avg
				best_y = y + window_size // 2
		
		# 텍스트 밀도가 너무 높으면 강제로 끝점 사용
		if min_density > 0.3:  # 임계값 조정 가능
			best_y = actual_end
			if self.debug: 
				print(f"⚠️ 안전한 분할점 없음, 강제 분할 at {best_y}")
		
		return best_y

	def enhanced_preprocess(self, image_path):
		"""한국어 OCR 정확도 향상을 위한 고급 전처리"""
		# PIL로 이미지 읽기
		img = Image.open(image_path)
		original_size = img.size
		
		# 1. 이미지 크기 조정 (더 정확한 기준)
		width, height = img.size
		target_height = 2400  # 한국어 OCR에 최적화된 높이 (높일수록 정확도 증가)
		if height < target_height:
			scale = target_height / height
			new_width = int(width * scale)
			new_height = int(height * scale)
			img = img.resize((new_width, new_height), Image.LANCZOS)
			if self.debug: 
				print(f"📏 크기 조정: {original_size} -> {img.size}")
		
		# 2. 그레이스케일 변환
		if img.mode != 'L':
			img = img.convert('L')
		
		# 3. 히스토그램 평활화 (조명 불균형 해결)
		img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_GRAY2BGR)
		img_cv_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
		clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8,8))  # 대비 강화
		img_clahe = clahe.apply(img_cv_gray)
		img = Image.fromarray(img_clahe)
		if self.debug: 
			print("📊 히스토그램 평활화 적용")
		
		# 4. 적응적 가우시안 블러 (이미지 품질에 따라 조정)
		blur_radius = self._calculate_optimal_blur(img)
		if blur_radius > 0:
			img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
			if self.debug: 
				print(f"🌀 블러 적용: radius={blur_radius}")
		
		# 5. 대비 강화
		enhancer = ImageEnhance.Contrast(img)
		img = enhancer.enhance(1.8)  # 대비 증가 (1.2 -> 1.8)
		
		# 6. RGB로 변환 (Surya 호환성)
		img = img.convert('RGB')
		
		if self.debug: 
			print("✅ 전처리 완료")
		return img

	def _calculate_optimal_blur(self, img):
		"""이미지 품질에 따른 최적 블러 반지름 계산"""
		img_array = np.array(img)
		
		# 라플라시안 분산으로 이미지 선명도 측정
		laplacian_var = cv2.Laplacian(img_array, cv2.CV_64F).var()
		
		if laplacian_var > 500:  # 매우 선명
			return 0.3
		elif laplacian_var > 200:  # 보통
			return 0.5
		elif laplacian_var > 100:  # 약간 흐림
			return 0.7
		else:  # 매우 흐림
			return 0  # 블러 적용 안함

	def _get_confidence_icon(self, confidence):
		"""신뢰도에 따른 시각적 아이콘 반환"""
		if confidence >= 0.9:
			return "🟢"  # 높은 신뢰도 (초록)
		elif confidence >= 0.7:
			return "🟡"  # 보통 신뢰도 (노랑)
		else:
			return "🔴"  # 낮은 신뢰도 (빨강)

	def fix_korean_ocr_errors(self, text):
		"""한국어 OCR 흔한 오인식 패턴 수정"""
		corrections = {
			r'(\d)\s+(\d)': r'\1\2',          # 숫자 사이 공백 제거
			r'(\d+,?\d*)乳': r'\g<1>원',       # 숫자 뒤 乳 -> 원
			r'2,000乳': '2,000원',            # 특정 케이스
			r'2\.000': '2,000',               # 2.000 -> 2,000
			r'(\d)\.(\d{3})': r'\1,\2',       # 숫자.000 -> 숫자,000
		}
		
		corrected_text = text
		for pattern, replacement in corrections.items():
			corrected_text = re.sub(pattern, replacement, corrected_text)
		
		return corrected_text

	def _merge_segment_results(self, segment_results):
		"""
		분할된 이미지들의 OCR 결과를 지능적으로 병합
		중복 텍스트 제거 (텍스트 기준!)
		"""
		if len(segment_results) == 1:
			text_lines, y_offset = segment_results[0]
			return [(line, y_offset) for line in text_lines]

		all_text_lines = []
		seen_texts = set()  # 이미 등장한 텍스트

		def normalize_text(text):
			# 기호, 공백, HTML 태그 제거 + 소문자 변환
			text = re.sub(r'<.*?>', '', text)
			text = re.sub(r'[\s·•\-–—,./\\(){}[\]|:;]', '', text)
			return text.strip().lower()

		for i, (text_lines, y_offset) in enumerate(segment_results):
			for line in text_lines:
				norm_text = normalize_text(line.text)
				# 이미 본 텍스트(혹은 포함된 텍스트)면 중복으로 간주
				if norm_text in seen_texts or any(norm_text in t or t in norm_text for t in seen_texts):
					if self.debug:
						print(f"🗑️ 중복 제거(텍스트기반): {line.text[:40]}")
					continue
				seen_texts.add(norm_text)
				all_text_lines.append((line, y_offset))
		
		if self.debug: 
			print(f"✅ 병합 완료: 총 {len(all_text_lines)}개 텍스트")
		return all_text_lines

	def _sort_by_position_with_segments(self, text_lines_with_offset):
		"""
		오프셋 정보를 고려한 읽기 순서 정렬 (절대 Y 기준 전체 정렬)
		"""
		if not text_lines_with_offset:
			return text_lines_with_offset

		# 각 텍스트라인의 절대 y_center와 x_center 계산
		abs_lines = []
		for line, y_offset in text_lines_with_offset:
			y_center = ((line.bbox[1] + line.bbox[3]) / 2) + y_offset
			x_center = (line.bbox[0] + line.bbox[2]) / 2
			abs_lines.append((line, y_offset, y_center, x_center))

		# 먼저 절대 y_center로 오름차순 정렬
		abs_lines.sort(key=lambda x: (x[2], x[3]))

		# 결과는 (line, y_offset)만 반환
		return [(line, y_offset) for line, y_offset, _, _ in abs_lines]

	def process_image_ocr(self, image_path):
		"""
		긴 이미지를 지능적으로 분할하여 OCR 처리
		
		Args:
			image_path: 이미지 경로
			
		Returns:
			OCR 결과 텍스트 라인들의 리스트
		"""
		print("🚀 스마트 이미지 OCR 처리 시작")
		print("="*50)
		
		# Surya 예측기 초기화
		self._init_predictors()
		
		# 1. 이미지 분할
		image_segments = self.smart_image_split(image_path)
		
		if len(image_segments) == 1:
			print("📄 단일 이미지 처리")
			
			# 기존 방식으로 처리
			processed_img = self.enhanced_preprocess(image_path)
			predictions = self.recognition_predictor([processed_img], det_predictor=self.detection_predictor)
			
			# 단일 이미지인 경우도 (text_line, offset) 형태로 변환
			if predictions and predictions[0].text_lines:
				final_text_lines_with_offset = [(line, 0) for line in predictions[0].text_lines]
			else:
				final_text_lines_with_offset = []
		
		else:
			print(f"📑 다중 세그먼트 처리 ({len(image_segments)}개)")
			
			# 2. 각 세그먼트별로 OCR 처리
			segment_results = []
			
			for i, (segment_img, y_offset) in enumerate(image_segments):
				print(f"\n🔍 세그먼트 {i+1}/{len(image_segments)} 처리 중...")
				
				# 세그먼트별 전처리
				# 임시 파일로 저장 후 전처리
				temp_path = f"temp_segment_{i}.png"
				segment_img.save(temp_path)
				
				try:
					processed_segment = self.enhanced_preprocess(temp_path)
					predictions = self.recognition_predictor([processed_segment], det_predictor=self.detection_predictor)
					
					if predictions and predictions[0].text_lines:
						segment_results.append((predictions[0].text_lines, y_offset))
						print(f"  ✅ {len(predictions[0].text_lines)}개 텍스트 검출")
					else:
						print("  ❌ 텍스트 없음")
						
				except Exception as e:
					print(f"  ❌ 세그먼트 처리 실패: {e}")
				
				# 임시 파일 정리
				if os.path.exists(temp_path):
					os.remove(temp_path)
			
			# 3. 결과 병합
			print(f"\n🔗 세그먼트 결과 병합 중...")
			final_text_lines_with_offset = self._merge_segment_results(segment_results)
		
		# 4. 최종 정렬 및 출력
		sorted_text_lines_with_offset = self._sort_by_position_with_segments(final_text_lines_with_offset)
		
		print(f"\n📄 최종 결과 ({len(sorted_text_lines_with_offset)}개 텍스트)")
		print("-" * 70)
		
		# 신뢰도 통계 계산
		confidences = []
		final_text_lines = []
		
		for i, (text_line, y_offset) in enumerate(sorted_text_lines_with_offset, 1):
			original_text = text_line.text
			corrected_text = self.fix_korean_ocr_errors(original_text)
			
			# 신뢰도 정보 가져오기
			confidence = getattr(text_line, 'confidence', None)
			if confidence is None:
				confidence = 1.0  # 신뢰도 정보가 없으면 기본값
			
			confidences.append(confidence)
			
			# 신뢰도에 따른 색상 표시 (이모지로)
			confidence_icon = self._get_confidence_icon(confidence)
			confidence_str = f"{confidence:.3f}"
			
			# 수정 여부 표시
			correction_mark = "📝" if original_text != corrected_text else "  "
			
			# 출력 형식: 번호. [신뢰도] 텍스트
			print(f"{i:3d}. {confidence_icon}[{confidence_str}] {correction_mark} {corrected_text}")
			
			# 최종 결과에는 원본 text_line 객체 추가
			final_text_lines.append(text_line)
		
		# 신뢰도 통계 출력
		if confidences:
			avg_confidence = sum(confidences) / len(confidences)
			min_confidence = min(confidences)
			max_confidence = max(confidences)
			low_confidence_count = sum(1 for c in confidences if c < 0.8)
			
			print(f"\n📊 신뢰도 통계")
			print(f"   평균: {avg_confidence:.3f} | 최소: {min_confidence:.3f} | 최대: {max_confidence:.3f}")
			print(f"   낮은 신뢰도 (<0.8): {low_confidence_count}개 ({low_confidence_count/len(confidences)*100:.1f}%)")
			
			# 신뢰도가 낮은 항목들 별도 표시
			if low_confidence_count > 0:
				print(f"\n⚠️ 검토가 필요한 낮은 신뢰도 항목들:")
				low_conf_items = []
				for i, (text_line, y_offset) in enumerate(sorted_text_lines_with_offset, 1):
					confidence = getattr(text_line, 'confidence', 1.0)
					if confidence < 0.8:
						corrected_text = self.fix_korean_ocr_errors(text_line.text)
						low_conf_items.append((i, confidence, corrected_text))
				
				# 신뢰도가 낮은 순으로 정렬
				low_conf_items.sort(key=lambda x: x[1])
				
				for line_num, conf, text in low_conf_items[:5]:  # 최대 5개만 표시
					print(f"   {line_num:3d}번: [{conf:.3f}] {text[:50]}...")
				
				if len(low_conf_items) > 5:
					print(f"   ... 외 {len(low_conf_items) - 5}개 더")
		
		print(f"\n💡 신뢰도 범례: 🟢높음(>0.9) 🟡보통(0.7-0.9) 🔴낮음(<0.7)")
		print(f"💡 수정 표시: 📝 = OCR 후 자동 수정됨")
		print(f"\n{'='*50}")
		print("🎉 스마트 OCR 처리 완료!")
		
		return final_text_lines

	def paddle_reocr_low_conf_lines(self, image_path, text_lines, conf_threshold=0.8, 
								   crop_margin=8, scale_factor=2.0, use_gpu=True):
		"""
		Surya 결과 중 신뢰도 conf_threshold 이하 줄만 해당 bbox 영역 crop→전처리→PaddleOCR 후 결과 비교.
		
		Args:
			image_path: 원본 이미지 경로
			text_lines: process_image_ocr의 반환값(list of text_line)
			conf_threshold: 신뢰도 임계값
			crop_margin: 크롭 시 추가 여백
			scale_factor: 이미지 확대 배율
			use_gpu: GPU 사용 여부
			
		Returns:
			비교 결과 리스트
		"""
		self._init_paddle_ocr(use_gpu)
		
		ori_img = cv2.imread(image_path)
		if ori_img is None:
			raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")

		results = []
		for idx, line in enumerate(text_lines, 1):
			conf = getattr(line, "confidence", 1.0)
			if conf > conf_threshold:
				continue
				
			bbox = getattr(line, "bbox", None)
			if not bbox:  # bbox 없는 줄은 스킵
				continue
				
			# bbox 정수화, margin 적용
			x1, y1, x2, y2 = map(int, bbox)
			x1 = max(0, x1 - crop_margin)
			y1 = max(0, y1 - crop_margin)
			x2 = min(ori_img.shape[1], x2 + crop_margin)
			y2 = min(ori_img.shape[0], y2 + crop_margin)
			
			crop_img = ori_img[y1:y2, x1:x2]
			if crop_img is None or crop_img.size == 0 or crop_img.shape[0] == 0 or crop_img.shape[1] == 0:
				if self.debug:
					print(f"⚠️ [경고] Crop 실패 - 라인 {idx}: bbox 좌표(x1={x1},y1={y1},x2={x2},y2={y2})가 비정상. 건너뜀.")
				continue
				
			temp_crop_path = f"temp_paddle_crop_{idx}.png"
			cv2.imwrite(temp_crop_path, crop_img)
			
			# 전처리(확대, 선명도 등)
			img = Image.open(temp_crop_path)
			width, height = img.size
			img = img.resize((int(width*scale_factor), int(height*scale_factor)), Image.LANCZOS)
			img_pil = ImageEnhance.Sharpness(img).enhance(1.5)
			img_pil = ImageEnhance.Contrast(img_pil).enhance(1.2)
			img_pil.save(temp_crop_path)
			
			# PaddleOCR 실행
			paddle_result = self.paddle_ocr.ocr(temp_crop_path, cls=True)
			
			# 결과 정리
			paddle_words = []
			for line_result in paddle_result:
				if line_result:
					for word_info in line_result:
						word, paddle_conf = word_info[1][0], word_info[1][1]
						paddle_words.append((word, paddle_conf))
			
			best_word, best_conf = ("", 0.0)
			if paddle_words:
				best_word, best_conf = max(paddle_words, key=lambda x: x[1])
			
			results.append({
				"idx": idx,
				"surya_text": line.text,
				"surya_conf": conf,
				"paddle_text": best_word,
				"paddle_conf": best_conf,
			})
			
			if self.debug:
				print(f"\n[{idx}] Surya({conf:.3f}): {line.text}")
				print(f"    Paddle({best_conf:.3f}): {best_word}")
			
			# 임시 파일 삭제
			if os.path.exists(temp_crop_path):
				os.remove(temp_crop_path)
		
		return results


# 사용 예시
if __name__ == "__main__":
	# OCR 프로세서 초기화
	ocr_processor = OCRProcessor(max_height=2000, overlap_ratio=0.1, debug=True)
	
	# 이미지 경로
	image_path = "/Users/parkjunseo/Downloads/박준서_졸업증명서.jpg"  # 여기에 실제 이미지 경로 입력

	# 이미지 파일이 존재하는지 확인
	if not os.path.exists(image_path):
		print(f"❌ 이미지 파일을 찾을 수 없습니다: {image_path}")
		print("현재 디렉토리의 이미지 파일을 사용하려면 image_path를 수정하세요.")
		import sys
		sys.exit(1)
	
	# OCR 처리
	result = ocr_processor.process_image_ocr(image_path)
	
	# 결과 출력
	print("\n최종 OCR 결과:")
	for idx, text_line in enumerate(result, 1):
		print(f"{idx}. {text_line.text} (신뢰도: {getattr(text_line, 'confidence', 1.0):.3f})")
	