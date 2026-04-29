from __future__ import annotations

from tao_medical_ai.domain.models import Disposition, FinalDecisionResponse, RiskLevel
from tao_medical_ai.domain.specialties import get_specialty_profile
from tao_medical_ai.infrastructure.llm.base import BaseExplainer


class OfflineExplainer(BaseExplainer):
    def explain(self, decision: FinalDecisionResponse) -> tuple[str, str]:
        profile = get_specialty_profile(decision.primary_domain)
        clinician = (
            f"service_line={decision.service_line}; domain={decision.primary_domain.value}; risk={decision.final_risk.value}; "
            f"disposition={decision.final_disposition.value}; human_review={decision.requires_human_review}; "
            f"intake_gaps={decision.intake_gaps or ['none']}; evidence_count={len(decision.evidence)}."
        )

        if decision.final_risk in {RiskLevel.high, RiskLevel.critical} or decision.final_disposition in {
            Disposition.emergency_department,
            Disposition.human_review,
        }:
            patient = (
                f"Luồng {profile.label} đánh giá đây là tình huống cần được kiểm tra trực tiếp sớm. "
                "Hãy đến cấp cứu hoặc liên hệ bác sĩ ngay, đặc biệt nếu triệu chứng đang kéo dài hoặc nặng hơn."
            )
        elif decision.final_risk == RiskLevel.moderate:
            patient = (
                f"Luồng {profile.label} chưa thấy mức nguy cơ cao nhất, nhưng chưa đủ an toàn để chỉ tự theo dõi hoàn toàn. "
                "Bạn nên được khám trực tiếp trong ngày hoặc theo thời gian đã khuyến nghị."
            )
        else:
            patient = (
                f"Luồng {profile.label} hiện xếp trường hợp này vào mức nguy cơ thấp trong mô hình hỗ trợ. "
                "Bạn vẫn cần theo dõi sát và đi khám nếu xuất hiện dấu hiệu báo động hoặc triệu chứng không cải thiện."
            )
        return clinician, patient
