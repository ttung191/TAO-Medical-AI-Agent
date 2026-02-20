import sys
import os
import json
import asyncio

# Thêm thư mục gốc vào path để import được app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.orchestrator import Orchestrator
from app.models.schemas import PatientCase

def load_golden_cases():
    with open("data/golden_cases.json", "r", encoding="utf-8") as f:
        return json.load(f)

def run_tests():
    orchestrator = Orchestrator()
    cases = load_golden_cases()
    
    print(f"🚀 STARTING EVALUATION ON {len(cases)} GOLDEN CASES...\n")
    
    passed = 0
    
    for test_case in cases:
        print(f"🔹 Testing Case {test_case['id']} ({test_case['category']})...")
        
        # Tạo object input
        p_case = PatientCase(
            case_id=str(test_case["id"]),
            symptoms=test_case["input"],
            medical_history=test_case.get("history", "")
        )
        
        # Chạy hệ thống
        result = orchestrator.process_case(p_case)
        
        # Kiểm tra kết quả
        final_diag = result.final_diagnosis
        expected = test_case["expected_result"]
        
        # Logic chấm điểm đơn giản
        is_correct_tier = final_diag.tier.name == expected["tier"] if final_diag else False
        keyword_hit = expected["diagnosis_keyword"].lower() in final_diag.diagnosis_summary.lower() if final_diag else False
        
        if keyword_hit: # Ưu tiên bắt đúng bệnh
            print(f"   ✅ PASS: Detected '{expected['diagnosis_keyword']}'")
            passed += 1
        else:
            print(f"   ❌ FAIL: Expected '{expected['diagnosis_keyword']}', got '{final_diag.diagnosis_summary if final_diag else 'None'}'")
            
        print("-" * 50)
        
    print(f"\n🏆 EVALUATION COMPLETE: {passed}/{len(cases)} Cases Passed.")

if __name__ == "__main__":
    run_tests()