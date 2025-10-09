import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from paddleocr import PaddleOCR
import os

def preprocess_image(image_path, scale_factor=2.0, save_processed=True):
    """
    이미지 전처리 함수
    - 이미지 확대
    - 노이즈 제거
    - 대비 및 선명도 향상
    - 그레이스케일 변환
    """
    
    # 1. 이미지 로드
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
    
    print(f"원본 이미지 크기: {img.shape[:2]}")
    
    # 2. 이미지 확대 (여러 방법 중 선택 가능)
    height, width = img.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    # INTER_CUBIC: 고품질 확대에 적합
    # INTER_LANCZOS4: 더 선명한 결과
    img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    print(f"확대된 이미지 크기: {img_resized.shape[:2]}")
    
    # 3. PIL로 변환하여 추가 처리
    img_pil = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    
    # 4. 선명도 향상
    enhancer = ImageEnhance.Sharpness(img_pil)
    img_sharp = enhancer.enhance(1.5)  # 1.0이 원본, 1.5는 50% 선명도 증가
    
    # 5. 대비 향상
    enhancer = ImageEnhance.Contrast(img_sharp)
    img_contrast = enhancer.enhance(1.3)  # 30% 대비 증가
    
    # 6. 밝기 조정 (필요시)
    enhancer = ImageEnhance.Brightness(img_contrast)
    img_bright = enhancer.enhance(1.1)  # 10% 밝기 증가
    
    # 7. 다시 OpenCV로 변환
    img_processed = cv2.cvtColor(np.array(img_bright), cv2.COLOR_RGB2BGR)
    
    # 8. 노이즈 제거 (가우시안 블러를 약하게 적용)
    img_denoised = cv2.GaussianBlur(img_processed, (3, 3), 0)
    
    # 9. 그레이스케일 변환
    img_gray = cv2.cvtColor(img_denoised, cv2.COLOR_BGR2GRAY)
    
    # 10. 적응적 임계값 적용 (선택사항 - 텍스트가 흐릿한 경우)
    img_thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # 11. 처리된 이미지 저장 (옵션)
    if save_processed:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        processed_path = f"{base_name}_processed.png"
        cv2.imwrite(processed_path, img_gray)
        print(f"전처리된 이미지 저장: {processed_path}")
        return processed_path, img_gray
    
    return None, img_gray

def advanced_preprocess_image(image_path, scale_factor=3.0):
    """
    고급 전처리 함수 (더 강력한 전처리)
    """
    img = cv2.imread(image_path)
    
    # 1. 큰 배율로 확대
    height, width = img.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    # 2. Super Resolution 기법 사용 (EDSR 모델 - 옵션)
    # 먼저 INTER_CUBIC으로 확대
    img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # 3. 언샤프 마스킹으로 선명도 향상
    gaussian = cv2.GaussianBlur(img_resized, (9, 9), 10.0)
    img_unsharp = cv2.addWeighted(img_resized, 1.5, gaussian, -0.5, 0)
    
    # 4. CLAHE (Contrast Limited Adaptive Histogram Equalization) 적용
    img_gray = cv2.cvtColor(img_unsharp, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_gray)
    
    # 5. 모폴로지 연산으로 텍스트 강화
    kernel = np.ones((2,2), np.uint8)
    img_morph = cv2.morphologyEx(img_clahe, cv2.MORPH_CLOSE, kernel)
    
    return img_morph

def run_ocr_with_preprocessing(image_path, use_advanced=False):
    """
    전처리와 OCR을 함께 실행하는 함수
    """
    
    # OCR 인스턴스 생성
    ocr = PaddleOCR(use_textline_orientation=True, lang='korean')
    
    print("=" * 50)
    print("원본 이미지 OCR 결과:")
    print("=" * 50)
    
    # 1. 원본 이미지로 OCR 실행
    original_result = ocr.predict(image_path)
    
    print("원본 OCR 결과:")
    # PaddleOCR 3.x 결과는 리스트 형식
    if isinstance(original_result, list):
        for item in original_result:
            if isinstance(item, dict):
                text = item.get('rec_text', '')
                score = item.get('rec_score', 0)
                print(f"텍스트: {text}, 신뢰도: {score:.2f}")
    
    print("\n" + "=" * 50)
    print("전처리된 이미지 OCR 결과:")
    print("=" * 50)
    
    # 2. 전처리된 이미지로 OCR 실행
    if use_advanced:
        processed_img = advanced_preprocess_image(image_path)
        # 임시 파일로 저장
        temp_path = "temp_advanced_processed.png"
        cv2.imwrite(temp_path, processed_img)
        processed_result = ocr.predict(temp_path)
        os.remove(temp_path)  # 임시 파일 삭제
    else:
        processed_path, _ = preprocess_image(image_path, scale_factor=2.5, save_processed=True)
        processed_result = ocr.predict(processed_path)
    
    print("전처리 후 OCR 결과:")
    # PaddleOCR 3.x 결과는 리스트 형식
    if isinstance(processed_result, list):
        for item in processed_result:
            if isinstance(item, dict):
                text = item.get('rec_text', '')
                score = item.get('rec_score', 0)
                print(f"텍스트: {text}, 신뢰도: {score:.2f}")
    
    return original_result, processed_result

# 메인 실행 부분
if __name__ == "__main__":
    # 이미지 경로 (샘플 이미지가 없으면 오류 발생)
    img_path = '/Users/parkjunseo/Downloads/박준서_졸업증명서.jpg'

    # 이미지 파일이 존재하는지 확인
    if not os.path.exists(img_path):
        print(f"❌ 이미지 파일을 찾을 수 없습니다: {img_path}")
        print("현재 디렉토리의 이미지 파일을 사용하려면 img_path를 수정하세요.")
        import sys
        sys.exit(1)
    
    try:
        # 기본 전처리로 OCR 실행
        print("기본 전처리 모드로 실행...")
        # original_result, processed_result = run_ocr_with_preprocessing(img_path, use_advanced=False)
        
        print("\n" + "=" * 50)
        print("고급 전처리 모드로 추가 실행...")
        print("=" * 50)
        
        # 고급 전처리로 OCR 실행 (선택사항)
        _, advanced_result = run_ocr_with_preprocessing(img_path, use_advanced=True)
        
    except Exception as e:
        print(f"오류 발생: {e}")
        print("필요한 라이브러리 설치:")
        print("pip install paddlepaddle paddleocr opencv-python pillow")

# 개별 전처리 함수들도 독립적으로 사용 가능
def simple_upscale_only(image_path, scale_factor=2.0):
    """단순 확대만 수행"""
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    output_path = f"upscaled_{scale_factor}x_{os.path.basename(image_path)}"
    cv2.imwrite(output_path, img_resized)
    print(f"확대된 이미지 저장: {output_path}")
    
    return output_path