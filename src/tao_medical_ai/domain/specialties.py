from __future__ import annotations

from dataclasses import dataclass

from tao_medical_ai.domain.models import Disposition, DomainName, RiskLevel, SpecialtyDescriptor


@dataclass(frozen=True)
class SpecialtyProfile:
    slug: DomainName
    label: str
    service_line: str
    description: str
    intended_use: str
    keywords: tuple[str, ...]
    required_intake_fields: tuple[str, ...]
    default_handoff_questions: tuple[str, ...]
    routine_next_steps: tuple[str, ...]
    urgent_next_steps: tuple[str, ...]
    critical_next_steps: tuple[str, ...]


SPECIALTY_PROFILES: dict[DomainName, SpecialtyProfile] = {
    DomainName.cardiology: SpecialtyProfile(
        slug=DomainName.cardiology,
        label="Cardiology Triage",
        service_line="cardiology-triage",
        description="Acute chest pain, palpitations, exertional symptoms, dyspnea with possible cardiac cause.",
        intended_use="Front-door triage and escalation support for potential acute cardiac presentations.",
        keywords=(
            "chest pain",
            "chest pressure",
            "palpitations",
            "syncope",
            "radiating pain",
            "diaphoresis",
            "exertional",
        ),
        required_intake_fields=(
            "timeline",
            "symptoms",
            "heart_rate",
            "systolic_bp",
            "spo2",
            "pain_character",
        ),
        default_handoff_questions=(
            "Khi nào cơn đau bắt đầu và có tăng khi gắng sức không?",
            "Đau có lan hàm, tay trái, lưng hoặc kèm vã mồ hôi không?",
            "Đã có ECG, men tim hoặc tiền sử bệnh tim mạch trước đó chưa?",
        ),
        routine_next_steps=(
            "Khám bác sĩ trong ngày nếu triệu chứng còn tái diễn.",
            "Theo dõi đau ngực, khó thở, ngất, vã mồ hôi.",
        ),
        urgent_next_steps=(
            "Đi đánh giá trực tiếp khẩn trong ngày, ưu tiên cơ sở có ECG và xét nghiệm tim mạch.",
            "Không tự lái xe nếu đang còn đau ngực hoặc chóng mặt.",
        ),
        critical_next_steps=(
            "Kích hoạt đánh giá cấp cứu ngay lập tức hoặc gọi cấp cứu nếu đang ở ngoài bệnh viện.",
            "Không trì hoãn để theo dõi tại nhà khi còn đau ngực, khó thở, ngất hoặc tụt huyết áp.",
        ),
    ),
    DomainName.neurology: SpecialtyProfile(
        slug=DomainName.neurology,
        label="Neurology Stroke Triage",
        service_line="neurology-stroke-triage",
        description="Focal neurological deficits, speech disturbance, facial droop, severe sudden headache, seizure-like events.",
        intended_use="Rapid screening and escalation support for suspected stroke or other focal neurologic emergencies.",
        keywords=(
            "weakness",
            "facial droop",
            "speech difficulty",
            "slurred speech",
            "numbness",
            "severe headache",
            "vision loss",
            "confusion",
            "seizure",
        ),
        required_intake_fields=(
            "timeline",
            "last_known_well",
            "symptoms",
            "systolic_bp",
            "glucose_if_available",
            "anticoagulant_history",
        ),
        default_handoff_questions=(
            "Thời điểm bình thường cuối cùng là khi nào?",
            "Có yếu liệt khu trú, méo miệng, nói khó hoặc mất thị lực không?",
            "Có đang dùng thuốc chống đông hay mới chấn thương đầu không?",
        ),
        routine_next_steps=(
            "Khám thần kinh sớm nếu triệu chứng nhẹ nhưng còn tái diễn.",
            "Theo dõi xuất hiện yếu liệt, méo miệng, nói khó, nhìn mờ, co giật.",
        ),
        urgent_next_steps=(
            "Đến cơ sở cấp cứu ngay trong ngày để loại trừ đột quỵ hoặc nguyên nhân thần kinh cấp.",
            "Không tự theo dõi tại nhà nếu còn triệu chứng khu trú hoặc lú lẫn.",
        ),
        critical_next_steps=(
            "Kích hoạt quy trình stroke/EMS ngay nếu có dấu FAST dương tính hoặc khởi phát cấp tính.",
            "Ghi nhận last-known-well rõ ràng và không trì hoãn chuyển viện.",
        ),
    ),
    DomainName.general_outpatient: SpecialtyProfile(
        slug=DomainName.general_outpatient,
        label="General Outpatient Assistant",
        service_line="general-outpatient-assistant",
        description="Low-to-moderate acuity outpatient symptom triage, follow-up planning, and safe next-step guidance.",
        intended_use="General ambulatory decision support when the case does not fit high-acuity cardiac or neurologic pathways.",
        keywords=(
            "sore throat",
            "cough",
            "rash",
            "abdominal pain",
            "nausea",
            "diarrhea",
            "fatigue",
            "medication refill",
        ),
        required_intake_fields=(
            "timeline",
            "symptoms",
            "temperature_c_if_fever",
            "hydration_status_if_gi_symptoms",
        ),
        default_handoff_questions=(
            "Triệu chứng kéo dài bao lâu và đang cải thiện hay xấu đi?",
            "Có sốt cao, mất nước, không ăn uống được hoặc đau tăng nhanh không?",
            "Bạn đang cần tự chăm sóc, hẹn khám gần, hay đánh giá khẩn?",
        ),
        routine_next_steps=(
            "Theo dõi triệu chứng và đặt hẹn khám ngoại trú nếu chưa cải thiện.",
            "Dùng hướng dẫn tự chăm sóc an toàn, quay lại khám nếu nặng hơn.",
        ),
        urgent_next_steps=(
            "Khám trong ngày hoặc urgent care nếu đau tăng, mất nước, sốt cao kéo dài hoặc chức năng giảm rõ.",
            "Không chỉ tự điều trị tại nhà khi thiếu dữ liệu quan trọng hoặc triệu chứng tiến triển.",
        ),
        critical_next_steps=(
            "Chuyển đánh giá cấp cứu ngay nếu có bất ổn sinh tồn hoặc dấu hiệu thần kinh/tim mạch nặng.",
            "Báo động human review trước khi xuất khuyến nghị cuối cùng.",
        ),
    ),
}


def get_specialty_profile(domain: DomainName) -> SpecialtyProfile:
    return SPECIALTY_PROFILES[domain]


def list_specialties() -> list[SpecialtyDescriptor]:
    return [
        SpecialtyDescriptor(
            slug=profile.slug,
            label=profile.label,
            description=profile.description,
            intended_use=profile.intended_use,
            required_intake_fields=list(profile.required_intake_fields),
            default_handoff_questions=list(profile.default_handoff_questions),
        )
        for profile in SPECIALTY_PROFILES.values()
    ]


def recommended_next_steps(domain: DomainName, risk: RiskLevel) -> list[str]:
    profile = get_specialty_profile(domain)
    if risk == RiskLevel.critical:
        return list(profile.critical_next_steps)
    if risk == RiskLevel.high:
        return list(profile.urgent_next_steps)
    return list(profile.routine_next_steps)


def default_disposition(domain: DomainName, risk: RiskLevel) -> Disposition:
    if risk == RiskLevel.critical:
        return Disposition.emergency_department
    if risk == RiskLevel.high:
        return Disposition.emergency_department if domain in {DomainName.cardiology, DomainName.neurology} else Disposition.urgent_care
    if risk == RiskLevel.moderate:
        return Disposition.primary_care if domain == DomainName.general_outpatient else Disposition.urgent_care
    return Disposition.self_care if domain == DomainName.general_outpatient else Disposition.primary_care
