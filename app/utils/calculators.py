def calculate_burch_wartofsky(symptoms: str, temp_c: float = 37.0) -> dict:
    """Tính điểm nguy cơ bão giáp dựa trên triệu chứng"""
    score = 0
    details = []
    text = symptoms.lower()

    # 1. Nhiệt độ (Demo logic)
    if temp_c >= 40: score += 30
    elif temp_c >= 38: score += 15

    # 2. Thần kinh
    if "coma" in text or "seizure" in text: 
        score += 30; details.append("CNS: Severe")
    elif "confused" in text or "agitated" in text or "delirium" in text: 
        score += 20; details.append("CNS: Moderate")

    # 3. Tiêu hóa
    if "jaundice" in text or "yellow" in text: 
        score += 20; details.append("GI: Jaundice")
    elif "vomiting" in text or "diarrhea" in text: 
        score += 10; details.append("GI: Mild")

    # 4. Tim mạch
    if "racing" in text or "pounding" in text: 
        score += 25; details.append("CV: Tachycardia")

    # Đánh giá
    result = "Unlikely"
    if score >= 45: result = "High Probability of Thyroid Storm"
    elif score >= 25: result = "Impending Storm"
    
    return {"score": score, "prediction": result, "details": details}