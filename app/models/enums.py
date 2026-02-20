from enum import Enum

class Tier(str, Enum):
    TIER_1 = "TIER_1_INITIAL"       # Sàng lọc / Đánh giá ban đầu
    TIER_2 = "TIER_2_SPECIALIST"    # Chuyên khoa sâu
    TIER_3 = "TIER_3_CONSULTANT"    # Hội chẩn cấp cao

class RiskLevel(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class AgentStatus(str, Enum):
    WORKING = "WORKING"
    COMPLETED = "COMPLETED"
    ESCALATED = "ESCALATED"
    RETURNED = "RETURNED"  # <-- MỚI: Trạng thái bị trả hồ sơ
    FAILED = "FAILED"

class EscalationDecision(str, Enum):
    RESOLVE = "RESOLVE"    # Tự giải quyết, không chuyển tiếp
    ESCALATE = "ESCALATE"  # Chuyển lên tuyến trên
    REJECT = "REJECT"      # Từ chối nhận, trả về tuyến dưới