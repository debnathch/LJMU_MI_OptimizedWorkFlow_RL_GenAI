import random

class ReportGeneratorStub:
    """
    Stub for LLM-based radiology report generation.
    """
    def __init__(self):
        self.templates = [
            "Patient shows signs of {finding}. Correlation with clinical history suggested.",
            "Visual inspection of {finding} indicates mild progression.",
            "No significant abnormalities detected regarding {finding}.",
            "Urgent follow-up required for {finding}."
        ]
        self.findings = ["Pneumonia", "Nodule", "Fracture", "Pleural Effusion"]

    def generate_report(self, case_id: int, is_urgent: bool):
        """
        Generates a mock report.
        """
        finding = random.choice(self.findings)
        template = random.choice(self.templates)
        report = template.format(finding=finding)
        
        if is_urgent:
            report = "URGENT PRELIMINARY REPORT: " + report
            
        return {
            "case_id": case_id,
            "report_text": report,
            "status": "Finalized" if not is_urgent else "Preliminary"
        }
