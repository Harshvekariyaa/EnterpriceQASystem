"""
generate_pdfs.py
Creates the knowledge base for the project: 10 policy PDFs of a FICTIONAL company
("NovaTech Solutions Pvt. Ltd."). Because the company is fictional, no LLM already
knows these facts, so every correct answer MUST come from retrieval (ideal for RAG tests).

Run:  python scripts/generate_pdfs.py
Output: data/pdfs/*.pdf
"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "pdfs"

# Same paragraph appears at the end of EVERY pdf on purpose.
# It gives our de-duplication step real duplicates to find and remove.
BOILERPLATE = (
    "Confidentiality Notice: This document is the property of NovaTech Solutions Pvt. Ltd. "
    "and is intended for internal use by employees only. Unauthorised copying, sharing or "
    "distribution outside the company is prohibited. For clarifications, contact the "
    "People and Culture team at hr@novatech.example."
)

DOCS = [
    {
        "file": "01_Leave_Policy.pdf",
        "title": "Leave Policy",
        "doc_id": "NT-HR-001", "dept": "Human Resources", "version": "3.2", "effective": "01 April 2025",
        "sections": [
            ("1. Purpose and Scope",
             ["This policy explains the types of leave available to all full-time employees of NovaTech Solutions. "
              "It applies to employees in all offices after they have completed onboarding. Interns and contractors "
              "are covered by their individual agreements."]),
            ("2. Annual (Earned) Leave",
             ["Every full-time employee receives 24 days of paid annual leave per calendar year. Leave is credited "
              "at 2 days per month. Employees may carry forward a maximum of 10 unused annual leave days to the next "
              "year; any balance above 10 days lapses on 31 December. Annual leave of more than 5 consecutive working "
              "days must be applied for at least 14 days in advance and approved by the reporting manager."]),
            ("3. Sick Leave",
             ["Employees are entitled to 12 days of paid sick leave per year. Sick leave cannot be carried forward "
              "or encashed. A medical certificate from a registered doctor is required if sick leave exceeds 2 "
              "consecutive days. Employees must inform their manager before 10:00 AM on the first day of sickness."]),
            ("4. Casual Leave",
             ["Each employee gets 8 days of casual leave per year for personal matters. A maximum of 3 casual leave "
              "days can be taken together at one time. Casual leave cannot be combined with sick leave."]),
            ("5. Maternity, Paternity and Parental Leave",
             ["Female employees are entitled to 26 weeks of paid maternity leave for the first two children. "
              "Male employees are entitled to 10 working days of paid paternity leave, which must be taken within "
              "3 months of the child's birth. Adoption leave of 12 weeks is available to the primary caregiver."]),
            ("6. Bereavement and Marriage Leave",
             ["Employees get 5 days of paid bereavement leave on the death of an immediate family member "
              "(parent, spouse, child or sibling). Employees receive 5 days of paid marriage leave for their own "
              "marriage, applied at least 30 days in advance."]),
            ("7. Public Holidays",
             ["NovaTech observes 12 public holidays each year. The list is published on the HR portal in the "
              "first week of January. Employees may also choose 2 optional (floating) holidays from a published list."]),
            ("8. Leave Without Pay",
             ["Leave without pay (LWP) may be approved by the department head for up to 30 days in a year when all "
              "paid leave has been used. LWP beyond 30 days requires approval from the HR Director."]),
        ],
    },
    {
        "file": "02_Remote_Work_Policy.pdf",
        "title": "Remote and Hybrid Work Policy",
        "doc_id": "NT-HR-002", "dept": "Human Resources", "version": "2.1", "effective": "01 July 2025",
        "sections": [
            ("1. Work Models",
             ["NovaTech follows a hybrid work model. Employees work from the office for 3 days per week (Tuesday, "
              "Wednesday and Thursday are the mandatory office days) and may work remotely on Monday and Friday. "
              "Fully remote work is allowed only for roles approved by the Vice President of the department."]),
            ("2. Eligibility",
             ["Employees become eligible for hybrid work after completing their 3-month probation period. "
              "Employees in their probation period must work from the office 5 days a week."]),
            ("3. Work From Anywhere (WFA)",
             ["Each employee may work from a location other than their home city for up to 20 days per calendar "
              "year under the Work From Anywhere programme. WFA must be inside India and requires manager approval "
              "at least 7 days in advance. Working from another country is not allowed under this policy because "
              "of tax and legal reasons."]),
            ("4. Core Working Hours",
             ["Core hours are 11:00 AM to 4:00 PM India Standard Time. Employees must be available on chat and "
              "video during core hours. Standard working hours are 8 hours per day, including a 1-hour lunch break."]),
            ("5. Home Office Allowance",
             ["Employees on the hybrid model receive a one-time home office setup allowance of Rs. 15,000 to buy a "
              "chair, desk or monitor. In addition, a monthly internet reimbursement of up to Rs. 1,000 is paid "
              "against a valid bill. The allowance can be claimed once every 3 years."]),
            ("6. Equipment and Security",
             ["Company laptops must be used for all work. Employees must connect through the company VPN when "
              "working remotely. Personal laptops are not permitted to access source code repositories."]),
        ],
    },
    {
        "file": "03_IT_Security_Policy.pdf",
        "title": "Information Technology Security Policy",
        "doc_id": "NT-IT-003", "dept": "Information Technology", "version": "4.0", "effective": "15 January 2025",
        "sections": [
            ("1. Passwords",
             ["Passwords must be at least 12 characters long and include uppercase letters, lowercase letters, "
              "numbers and one special character. Passwords must be changed every 90 days. The last 5 passwords "
              "cannot be reused. Employees must never share passwords, even with IT staff."]),
            ("2. Multi-Factor Authentication",
             ["Multi-factor authentication (MFA) is mandatory for email, VPN, the code repository and the cloud "
              "console. The approved MFA method is the NovaTech Authenticator app. SMS-based codes are not allowed."]),
            ("3. Device Security",
             ["All company laptops have full-disk encryption enabled. Screens lock automatically after 5 minutes of "
              "inactivity. USB storage devices are blocked by default; exceptions need written approval from the "
              "IT Security team. Lost or stolen devices must be reported to the IT Helpdesk within 2 hours."]),
            ("4. Incident Reporting",
             ["Any suspected security incident, such as a phishing email or malware, must be reported to "
              "security@novatech.example immediately. The Security Operations Centre will acknowledge reports "
              "within 30 minutes. Critical incidents are resolved or escalated within 4 hours."]),
            ("5. Software Installation",
             ["Only software from the approved software catalogue may be installed. Requests for new software are "
              "raised through the IT Service Portal and are reviewed within 3 working days."]),
            ("6. Data Classification",
             ["Company data is classified into four levels: Public, Internal, Confidential and Restricted. "
              "Restricted data, such as customer financial records, must never be stored on personal devices or "
              "personal cloud storage."]),
            ("7. Security Awareness Training",
             ["Every employee must complete the annual security awareness training within 30 days of joining and "
              "once every year after that. Employees who fail the phishing simulation test twice must repeat the "
              "training."]),
        ],
    },
    {
        "file": "04_Expense_Reimbursement_Policy.pdf",
        "title": "Expense Reimbursement Policy",
        "doc_id": "NT-FIN-004", "dept": "Finance", "version": "2.4", "effective": "01 April 2025",
        "sections": [
            ("1. General Rules",
             ["Employees can claim reimbursement for business expenses that are reasonable, necessary and supported "
              "by original bills or receipts. Personal expenses are never reimbursable. Claims must be submitted "
              "through the Expense Portal within 30 days of the expense date; claims submitted later are rejected."]),
            ("2. Approval Limits",
             ["Claims up to Rs. 5,000 are approved by the reporting manager. Claims between Rs. 5,001 and Rs. 25,000 "
              "need approval from the department head. Claims above Rs. 25,000 need approval from the Finance "
              "Controller. Receipts are mandatory for any single expense above Rs. 500."]),
            ("3. Meals and Client Entertainment",
             ["Employees on business duty may claim meals up to Rs. 800 per day. Client entertainment expenses are "
              "capped at Rs. 2,500 per person per event and must have prior approval from the department head. "
              "Alcohol is not reimbursable under any circumstances."]),
            ("4. Mobile and Internet",
             ["Employees in client-facing roles may claim up to Rs. 600 per month for mobile phone bills. "
              "Internet reimbursement for remote work is covered in the Remote and Hybrid Work Policy."]),
            ("5. Professional Development",
             ["Each employee has an annual learning budget of Rs. 30,000 for courses, certifications and books. "
              "Courses costing more than Rs. 30,000 need written approval from the department head. If an employee "
              "leaves within 12 months of completing a company-paid certification, the cost must be repaid."]),
            ("6. Payment Timeline",
             ["Approved claims are paid with the next salary cycle if approved before the 20th of the month. "
              "Claims approved after the 20th are paid in the following month. Payments are made to the employee's "
              "registered salary bank account."]),
        ],
    },
    {
        "file": "05_Travel_Policy.pdf",
        "title": "Business Travel Policy",
        "doc_id": "NT-FIN-005", "dept": "Finance", "version": "1.8", "effective": "01 June 2025",
        "sections": [
            ("1. Booking Travel",
             ["All business travel must be booked through the company travel desk using the Travel Portal. "
              "Travel requests must be raised at least 7 days before domestic travel and 21 days before "
              "international travel. Last-minute bookings need written approval from the department head."]),
            ("2. Air Travel",
             ["Economy class is mandatory for domestic flights and for international flights shorter than 6 hours. "
              "Employees at Director level and above, and any employee on an international flight longer than "
              "6 hours, may fly Premium Economy. Business class is allowed only with approval from the CEO."]),
            ("3. Train and Road Travel",
             ["For train journeys, employees may travel in AC 2-Tier. Local travel by taxi or ride-hailing is "
              "reimbursed with receipts. If a personal vehicle is used for business, the company pays Rs. 12 per "
              "kilometre for cars and Rs. 5 per kilometre for two-wheelers."]),
            ("4. Hotel Accommodation",
             ["Hotel limits per night are: Rs. 6,000 in Tier-1 cities (Mumbai, Delhi, Bengaluru, Chennai, "
              "Hyderabad, Kolkata), Rs. 4,000 in other Indian cities, and USD 180 for international travel. "
              "Hotels must be booked through the travel desk where possible."]),
            ("5. Per Diem",
             ["A daily travel allowance (per diem) of Rs. 1,200 is paid for domestic travel and USD 50 for "
              "international travel. This allowance covers meals and small incidental expenses; separate meal "
              "claims cannot be made on the same day."]),
            ("6. Travel Insurance",
             ["All international business travel is covered by the company's group travel insurance. Employees must "
              "download the insurance certificate from the Travel Portal before departure."]),
        ],
    },
    {
        "file": "06_Code_of_Conduct.pdf",
        "title": "Code of Conduct and Ethics",
        "doc_id": "NT-LEG-006", "dept": "Legal and Compliance", "version": "5.0", "effective": "01 January 2025",
        "sections": [
            ("1. Our Values",
             ["NovaTech's four core values are Integrity, Customer Focus, Ownership and Respect. Every employee "
              "is expected to follow these values in all dealings with colleagues, customers and partners."]),
            ("2. Anti-Harassment",
             ["NovaTech has zero tolerance for harassment, discrimination or bullying. Complaints of sexual "
              "harassment are handled by the Internal Complaints Committee (ICC). The ICC must complete its inquiry "
              "within 90 days of receiving a complaint. Retaliation against anyone who files a complaint in good "
              "faith is strictly prohibited."]),
            ("3. Conflict of Interest",
             ["Employees must disclose any situation where personal interests may conflict with the company's "
              "interests. Employees may not work for a competitor or take a second job without written approval "
              "from the HR Director. Disclosure must be made through the Compliance Portal within 15 days of "
              "the conflict arising."]),
            ("4. Gifts and Hospitality",
             ["Employees may accept gifts from vendors or clients only if the value is below Rs. 2,000. Any gift "
              "worth Rs. 2,000 or more must be declined or reported to the Compliance team within 7 days. "
              "Cash gifts and gift cards must never be accepted."]),
            ("5. Whistleblower Protection",
             ["Employees can report unethical behaviour confidentially through the Ethics Hotline at "
              "ethics@novatech.example or by calling 1800-000-0000, available 24 hours a day. Reports can be "
              "made anonymously. The Ethics Committee reviews every report and responds within 10 working days."]),
            ("6. Consequences of Violations",
             ["Violations of this code may lead to disciplinary action ranging from a written warning to "
              "termination of employment, depending on how serious the violation is."]),
        ],
    },
    {
        "file": "07_Employee_Onboarding_Guide.pdf",
        "title": "Employee Onboarding Guide",
        "doc_id": "NT-HR-007", "dept": "Human Resources", "version": "2.0", "effective": "01 March 2025",
        "sections": [
            ("1. Before Day One",
             ["New joiners receive an offer pack by email 7 days before the joining date. The pack contains the "
              "documents checklist: identity proof, address proof, education certificates, previous employment "
              "relieving letter and 2 passport-size photographs. Laptops are shipped to remote joiners 3 days "
              "before they start."]),
            ("2. First Day",
             ["Orientation begins at 10:00 AM at the office reception. On day one, IT issues the laptop, company "
              "email ID and access card. Every new employee is paired with a buddy who guides them for the first "
              "30 days."]),
            ("3. First 30, 60 and 90 Days",
             ["During the first 30 days, employees complete mandatory training on security, the Code of Conduct "
              "and POSH (Prevention of Sexual Harassment). At 60 days, the manager conducts a feedback "
              "conversation. At 90 days, a probation review decides confirmation of employment."]),
            ("4. Probation",
             ["The standard probation period is 3 months. It can be extended once by up to 3 additional months if "
              "performance expectations are not met. During probation, the notice period is 15 days for both "
              "sides. After confirmation, the notice period is 60 days for employees and 90 days for managers "
              "and above."]),
            ("5. Payroll",
             ["Salaries are credited on the last working day of each month. Payslips are available on the HR "
              "Portal. Employees must submit investment declarations for tax savings by 15 January each year."]),
            ("6. Who to Contact",
             ["For IT problems, contact the IT Helpdesk at helpdesk@novatech.example or extension 4000. For "
              "payroll queries, contact payroll@novatech.example. For HR questions, write to hr@novatech.example."]),
        ],
    },
    {
        "file": "08_Performance_Review_Policy.pdf",
        "title": "Performance Review Policy",
        "doc_id": "NT-HR-008", "dept": "Human Resources", "version": "1.5", "effective": "01 April 2025",
        "sections": [
            ("1. Review Cycle",
             ["NovaTech runs two review cycles each year: the mid-year review in October and the annual review "
              "in April. The performance year runs from April to March. Goals must be set within the first "
              "30 days of the performance year."]),
            ("2. Rating Scale",
             ["Employees are rated on a 5-point scale: 1 - Needs Improvement, 2 - Partially Meets Expectations, "
              "3 - Meets Expectations, 4 - Exceeds Expectations, 5 - Outstanding. Ratings are decided after a "
              "calibration meeting of department heads to keep ratings fair across teams."]),
            ("3. 360-Degree Feedback",
             ["Employees at Senior Manager level and above receive 360-degree feedback from at least 5 peers, "
              "direct reports and stakeholders. Feedback is anonymous and is shared in summarised form."]),
            ("4. Salary Increments",
             ["The annual increment depends on the rating. A rating of 3 gives a typical increment of 8 percent, "
              "a rating of 4 gives 12 percent, and a rating of 5 gives 15 percent. Employees with a rating of 1 "
              "or 2 are not eligible for an increment. Increments take effect from 1 May."]),
            ("5. Performance Improvement Plan",
             ["Employees rated 1 are placed on a Performance Improvement Plan (PIP) of 60 days. The manager and "
              "HR review progress every 2 weeks. If the targets are not met by the end of the PIP, the employee "
              "may be exited as per company policy."]),
            ("6. Promotions",
             ["Promotions are considered once a year in April. An employee must have at least 18 months in the "
              "current role and a rating of 4 or above in the last review cycle to be eligible."]),
        ],
    },
    {
        "file": "09_Data_Privacy_Policy.pdf",
        "title": "Data Privacy and Protection Policy",
        "doc_id": "NT-LEG-009", "dept": "Legal and Compliance", "version": "2.2", "effective": "01 February 2025",
        "sections": [
            ("1. Purpose",
             ["This policy explains how NovaTech collects, uses, stores and protects personal data of customers "
              "and employees, in line with India's Digital Personal Data Protection Act, 2023."]),
            ("2. Data Collection Principles",
             ["NovaTech collects only the personal data that is necessary for a stated purpose and obtains consent "
              "before collection. Personal data is never sold to third parties."]),
            ("3. Data Retention",
             ["Customer personal data is retained for 5 years after the end of the contract and then securely "
              "deleted. Employee records are retained for 7 years after an employee leaves. Recruitment data of "
              "unsuccessful candidates is deleted after 12 months. CCTV footage is stored for 30 days."]),
            ("4. Data Subject Requests",
             ["Customers and employees can request access to, correction of, or deletion of their personal data by "
              "writing to privacy@novatech.example. The Data Protection Officer responds to all requests within "
              "30 days."]),
            ("5. Data Breach Response",
             ["If a personal data breach occurs, the Data Protection Officer must be informed within 24 hours of "
              "discovery. Affected individuals and the Data Protection Board of India are notified within 72 hours "
              "when the breach is likely to cause harm."]),
            ("6. Third-Party Vendors",
             ["Vendors that process personal data on NovaTech's behalf must sign a Data Processing Agreement and "
              "pass a security assessment before they are onboarded. Vendors are re-assessed every 12 months."]),
        ],
    },
    {
        "file": "10_Employee_Benefits_Guide.pdf",
        "title": "Employee Benefits Guide",
        "doc_id": "NT-HR-010", "dept": "Human Resources", "version": "3.0", "effective": "01 April 2025",
        "sections": [
            ("1. Health Insurance",
             ["All employees are covered by a group health insurance policy with a sum insured of Rs. 5,00,000 "
              "per family. The policy covers the employee, spouse, up to 2 children and dependent parents. "
              "Employee and spouse coverage is fully paid by NovaTech. Parents can be added by paying an "
              "additional premium of Rs. 8,000 per year per parent."]),
            ("2. Life and Accident Insurance",
             ["Every employee is covered by group term life insurance equal to 4 times the annual salary. "
              "Accidental death and disability cover of Rs. 10,00,000 is also provided at no cost."]),
            ("3. Provident Fund and Gratuity",
             ["Both the employee and NovaTech contribute 12 percent of basic salary to the Provident Fund every "
              "month. Gratuity is payable after 5 years of continuous service, as per the Payment of Gratuity Act."]),
            ("4. Wellness Programme",
             ["Each employee receives a yearly wellness allowance of Rs. 12,000 that can be used for gym "
              "membership, yoga classes or health check-ups. A free annual health check-up is provided to employees "
              "above 35 years of age. The Employee Assistance Programme offers free confidential counselling, "
              "available 24 hours a day."]),
            ("5. Meal and Transport",
             ["The office cafeteria offers subsidised meals at Rs. 50 per meal. Employees who work late shifts "
              "after 8:00 PM are provided cab facility to their home within a 25 km radius."]),
            ("6. Referral Bonus",
             ["Employees who refer a candidate who is hired and completes 6 months receive a referral bonus of "
              "Rs. 25,000. For senior roles at Manager level and above, the bonus is Rs. 50,000."]),
            ("7. Long Service Awards",
             ["NovaTech recognises employees on completing 5, 10 and 15 years of service with an award and a "
              "cash gift of Rs. 10,000, Rs. 25,000 and Rs. 50,000 respectively."]),
        ],
    },
]


def build_pdf(doc: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / doc["file"]
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=18, spaceAfter=8)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12.5, spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=10.5, leading=15, spaceAfter=6)
    small = ParagraphStyle("small", parent=body, fontSize=9, textColor=colors.grey)

    pdf = SimpleDocTemplate(
        str(path), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title=doc["title"], author="NovaTech Solutions Pvt. Ltd.",
        subject=f"{doc['dept']} | {doc['doc_id']} | v{doc['version']}",
    )
    story = [Paragraph("NovaTech Solutions Pvt. Ltd.", small), Paragraph(doc["title"], h1)]

    meta = [
        ["Document ID", doc["doc_id"], "Department", doc["dept"]],
        ["Version", doc["version"], "Effective Date", doc["effective"]],
    ]
    table = Table(meta, colWidths=[3 * cm, 4 * cm, 3.2 * cm, 5 * cm])
    table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
        ("BACKGROUND", (2, 0), (2, -1), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
    ]))
    story += [table, Spacer(1, 10)]

    for heading, paragraphs in doc["sections"]:
        story.append(Paragraph(heading, h2))
        for p in paragraphs:
            story.append(Paragraph(p, body))

    story += [Spacer(1, 12), Paragraph("Document Notice", h2), Paragraph(BOILERPLATE, body)]
    pdf.build(story)
    return path


if __name__ == "__main__":
    for d in DOCS:
        print("created:", build_pdf(d).name)
