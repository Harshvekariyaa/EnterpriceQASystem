# data.py  (Step 9 - part 1)
# Job: build the labelled dataset for the in-scope / out-of-scope classifier.
#   label 1 = IN-SCOPE  : the 10 NovaTech policy documents can answer it
#   label 0 = OUT-OF-SCOPE : they cannot answer it
#
# Questions are written in GROUPS of 3 (same meaning, different words).
# Rule: all questions of one group stay in the SAME split (train / validation / test).
# If they were mixed, the model would see a rephrased copy of a test question while training
# = "data leakage", and the test score would look better than the truth.

import csv
import random

# category "policy"    -> label 1
# category "general"   -> label 0 : nothing to do with the company
# category "near_miss" -> label 0 : SOUNDS like a company question, but the documents do not answer it
GROUPS = [
    # ------------------------------ IN-SCOPE (policy) ------------------------------
    ("policy", ["How many sick days am I allowed in a year?", "What is the yearly sick leave entitlement?", "How many paid days off can I take when I am unwell?"]),
    ("policy", ["How many days of leave can new fathers take?", "What is the paternity leave policy?", "Am I entitled to time off after my wife has a baby?"]),
    ("policy", ["How many casual leave days are there?", "Can I take 5 casual leave days in a row?", "What is casual leave meant for?"]),
    ("policy", ["What happens to my unused annual leave at year end?", "Does annual leave lapse on December 31?", "Can I carry over my earned leave to next year?"]),
    ("policy", ["How many days off do I get when a parent passes away?", "Is there leave for my own wedding?", "What is the bereavement leave entitlement?"]),
    ("policy", ["How many public holidays does NovaTech observe?", "Where can I find the holiday list?", "Are there optional holidays I can choose?"]),
    ("policy", ["How many days a week must I come to the office?", "Can I work from home on Fridays?", "What is the hybrid work schedule?"]),
    ("policy", ["Can I work from a different country for a month?", "What is the Work From Anywhere programme?", "How do I get approval to work from my hometown?"]),
    ("policy", ["Does the company pay for my internet bill?", "Is there money to buy an office chair for home?", "How often can I claim the home office allowance?"]),
    ("policy", ["Can probationers work remotely?", "I am new; can I work from home during probation?", "Who is allowed to be fully remote?"]),
    ("policy", ["What are the password complexity requirements?", "Can I reuse an old password?", "How frequently must I update my password?"]),
    ("policy", ["Which authenticator app does the company use?", "Is multi-factor authentication mandatory for email?", "Do I need two-step verification for the VPN?"]),
    ("policy", ["Can I plug a USB drive into my work laptop?", "What should I do if my laptop is stolen?", "How long before my screen locks automatically?"]),
    ("policy", ["Who do I tell if I get a phishing email?", "How quickly does the security team respond to reports?", "Where do I report malware on my machine?"]),
    ("policy", ["How do I request new software?", "Can I install my own apps on the company laptop?", "How long does software approval take?"]),
    ("policy", ["What are the data classification levels?", "When must I finish the security awareness training?", "Can I save restricted customer data on my personal cloud?"]),
    ("policy", ["Who approves expenses above 5000 rupees?", "What is the approval limit for my manager?", "Which expense claims go to the department head?"]),
    ("policy", ["What is the daily meal allowance on business trips?", "How much can I spend on client entertainment per person?", "Are drinks reimbursed on client dinners?"]),
    ("policy", ["How much is the learning allowance for courses?", "Do I have to repay my certification if I resign?", "Can I use company money to buy technical books?"]),
    ("policy", ["When will my approved claim be paid?", "Is my mobile phone bill reimbursed?", "How are expense reimbursements paid out?"]),
    ("policy", ["How early should I request domestic travel?", "Where do I book business flights?", "Do I need approval for last-minute bookings?"]),
    ("policy", ["Which flight class am I entitled to?", "When can I fly premium economy?", "What class of train travel is allowed?"]),
    ("policy", ["What is the hotel budget in Delhi?", "How much daily allowance do I get for international travel?", "What is the nightly hotel cap abroad?"]),
    ("policy", ["How much per km is paid for using a two-wheeler?", "Is there insurance for overseas trips?", "Where do I download my travel insurance certificate?"]),
    ("policy", ["What are the company's core values?", "Who handles sexual harassment complaints?", "What is the rule about retaliation for complaints?"]),
    ("policy", ["Can I take a second job?", "What should I do with a gift from a client?", "Can I accept gift cards from vendors?"]),
    ("policy", ["How do I report misconduct confidentially?", "Is there an ethics hotline?", "How quickly does the ethics committee respond?"]),
    ("policy", ["What happens if I violate the code of conduct?", "What disciplinary actions are possible?", "Could I be fired for breaking the code?"]),
    ("policy", ["What documents do I bring on joining day?", "When do new joiners receive the offer pack?", "When is a laptop shipped to remote joiners?"]),
    ("policy", ["What time does orientation start?", "Will I get a buddy when I join?", "What do I receive on my first day?"]),
    ("policy", ["Can probation be extended?", "What is the notice period for managers?", "What is the notice period during probation?"]),
    ("policy", ["Where can I download my payslip?", "By when must I submit investment declarations?", "On which day is salary credited?"]),
    ("policy", ["What does a rating of 5 mean?", "How are performance ratings defined?", "What is the scale used for performance reviews?"]),
    ("policy", ["What increment will I get with a rating of 3?", "Are employees with low ratings eligible for a raise?", "When do salary increments take effect?"]),
    ("policy", ["What happens if I am put on a PIP?", "Who receives 360-degree feedback?", "How often are PIP progress reviews held?"]),
    ("policy", ["What are the promotion eligibility rules?", "How long must I stay in a role before promotion?", "When are promotions considered?"]),
    ("policy", ["How long are employee records kept after leaving?", "How long is recruitment data of rejected candidates retained?", "What is the data retention period for customers?"]),
    ("policy", ["How do I ask for my personal data to be deleted?", "Who is the Data Protection Officer?", "How long does the company take to respond to a data request?"]),
    ("policy", ["Who must be told when a data breach happens?", "Within what time are affected people notified of a breach?", "Do vendors need to sign a data processing agreement?"]),
    ("policy", ["Can I add my parents to the health insurance?", "Who is covered by the group health policy?", "How much extra premium is charged for each parent?"]),
    ("policy", ["How much life insurance cover do I get?", "What is the PF contribution percentage?", "After how many years is gratuity payable?"]),
    ("policy", ["What is the yearly wellness allowance for?", "Is a free health check-up available?", "Does the company provide counselling support?"]),
    ("policy", ["How much does a cafeteria meal cost?", "Is a cab provided for late shifts?", "What cash gift is given for 10 years of service?"]),
    ("policy", ["How much is the referral bonus for managers?", "When do I receive the referral bonus?", "What are the rules for employee referrals?"]),

    # ------------------------------ OUT-OF-SCOPE (general) ------------------------------
    ("general", ["What is the capital of Australia?", "Which city is the capital of Japan?", "Name the capital of Canada."]),
    ("general", ["Who wrote Romeo and Juliet?", "Who is the author of Pride and Prejudice?", "Which author created Sherlock Holmes?"]),
    ("general", ["How tall is Mount Everest?", "What is the longest river in the world?", "Which is the largest ocean on Earth?"]),
    ("general", ["How do plants make food from sunlight?", "What is the process of photosynthesis?", "Why are leaves green?"]),
    ("general", ["Who was the first prime minister of India?", "When did India become independent?", "Who led the Salt March?"]),
    ("general", ["What is the speed of light?", "How fast does light travel?", "What is the boiling point of water in Fahrenheit?"]),
    ("general", ["Who won the last football world cup?", "Which country has won the most Olympic gold medals?", "Who is the top scorer in IPL history?"]),
    ("general", ["When is the next cricket match of India?", "How many players are in a basketball team?", "What are the rules of tennis scoring?"]),
    ("general", ["How do I make paneer butter masala?", "What is the recipe for chocolate brownies?", "How long should I boil pasta?"]),
    ("general", ["How can I make fluffy pancakes?", "What is the best way to cook rice?", "How do I prepare masala chai?"]),
    ("general", ["How do I reverse a string in Python?", "How do I sort a list in JavaScript?", "What is a Python decorator?"]),
    ("general", ["How do I fix a null pointer exception in Java?", "What is the difference between SQL and NoSQL?", "How do I centre a div in CSS?"]),
    ("general", ["What is a neural network?", "Explain gradient descent.", "What is the difference between AI and machine learning?"]),
    ("general", ["How does blockchain work?", "What is cloud computing?", "Explain how the internet works."]),
    ("general", ["Should I invest in gold or property?", "How does compound interest work?", "Which mutual fund gives the best returns?"]),
    ("general", ["What is the current price of gold?", "What is the dollar to rupee rate today?", "How is the stock market doing today?"]),
    ("general", ["How can I lose weight quickly?", "What are the symptoms of the flu?", "How many hours of sleep do adults need?"]),
    ("general", ["What exercises strengthen the back?", "Is turmeric good for inflammation?", "How do I lower blood pressure naturally?"]),
    ("general", ["What are the best places to visit in Goa?", "Which is the cheapest month to fly to Paris?", "Do I need a visa for Thailand?"]),
    ("general", ["What is the weather like in Delhi in July?", "Will it rain in Mumbai tomorrow?", "What is the temperature in Surat today?"]),
    ("general", ["Who directed the movie Inception?", "What are the best web series to watch?", "Recommend a good book to read."]),
    ("general", ["Tell me a funny joke.", "Can you sing a song?", "Tell me an interesting fact."]),
    ("general", ["How are you today?", "What is your name?", "Who created you?"]),
    ("general", ["Which laptop should I buy for gaming?", "What is the best phone under 20000 rupees?", "How do I choose a good pair of running shoes?"]),
    ("general", ["How do I learn guitar quickly?", "What is the best way to learn French?", "How can I improve my English speaking?"]),
    ("general", ["How do I apply for a passport?", "How do I renew my driving licence?", "How can I link Aadhaar with PAN?"]),
    ("general", ["How do I write a good resume?", "What are common interview questions?", "How do I prepare for a job interview?"]),
    ("general", ["What is the meaning of the word ephemeral?", "Translate good morning into Spanish.", "What is the plural of cactus?"]),
    ("general", ["Solve 15 multiplied by 12.", "What is the square root of 144?", "What is 20 percent of 350?"]),
    ("general", ["What is the time in London now?", "What is today's date?", "How many days are there in a leap year?"]),

    # ------------------------------ OUT-OF-SCOPE (near-miss) ------------------------------
    ("near_miss", ["What is the leave policy at Microsoft?", "How many sick days does Amazon give its employees?", "What is the maternity leave policy at Infosys?"]),
    ("near_miss", ["What is TCS's work from home policy?", "Does Google allow remote work?", "What is Wipro's hybrid policy?"]),
    ("near_miss", ["Who founded NovaTech Solutions?", "Who is the managing director of NovaTech?", "Who is the head of HR at NovaTech?"]),
    ("near_miss", ["Where is the NovaTech headquarters?", "What was NovaTech's revenue last year?", "How many offices does NovaTech have?"]),
    ("near_miss", ["What is the salary band for a senior developer at NovaTech?", "How much is the joining bonus at NovaTech?", "What is the CTC of a fresher at NovaTech?"]),
    ("near_miss", ["Does NovaTech provide free parking?", "Is there a gym in the NovaTech office?", "Does NovaTech give employees a free laptop bag?"]),
    ("near_miss", ["Are jeans allowed at NovaTech on weekdays?", "Is there a formal dress code for client meetings?", "Can I wear casual clothes on Fridays?"]),
    ("near_miss", ["Are NovaTech employees given ESOPs?", "Can employees buy company shares?", "Is there a stock purchase plan at NovaTech?"]),
    ("near_miss", ["What is NovaTech's policy on pets in the office?", "Can I bring my dog to work?", "What is the sabbatical policy?"]),
    ("near_miss", ["What products does NovaTech sell?", "Which clients does NovaTech work with?", "What technology stack does NovaTech use?"]),
    ("near_miss", ["What time does the cafeteria open?", "Which dishes are served in the cafeteria?", "Is the cafeteria open on weekends?"]),
    ("near_miss", ["How do I apply for a US business visa?", "What are the working hours at the Bengaluru office?", "How do I reach the Pune office from the airport?"]),
    ("near_miss", ["What is the legal retirement age in India?", "What does the labour law say about overtime pay?", "What is the minimum wage in Gujarat?"]),
    ("near_miss", ["How do I reset my Wi-Fi router?", "How do I fix a blue screen error on Windows?", "What is the Wi-Fi password in the office?"]),
]


def build_rows():
    """Turn GROUPS into a list of rows: question, label, group_id, category."""
    rows = []
    group_id = 0
    for category, questions in GROUPS:
        group_id = group_id + 1
        label = 1 if category == "policy" else 0
        for question in questions:
            rows.append({"question": question, "label": label,
                         "group_id": group_id, "category": category})
    return rows


def assign_splits(rows, seed, fractions=(0.70, 0.15, 0.15)):
    """Give every row a split: train / validation / test.
    Whole GROUPS are moved together. Each category is split separately,
    so near-miss questions appear in every split."""
    random.seed(seed)
    groups_by_category = {}
    for row in rows:
        groups_by_category.setdefault(row["category"], set()).add(row["group_id"])

    split_of_group = {}
    for category in sorted(groups_by_category):
        group_ids = sorted(groups_by_category[category])
        random.shuffle(group_ids)
        total = len(group_ids)
        n_val = max(1, round(total * fractions[1]))
        n_test = max(1, round(total * fractions[2]))
        n_train = total - n_val - n_test
        for position, group_id in enumerate(group_ids):
            if position < n_train:
                split_of_group[group_id] = "train"
            elif position < n_train + n_val:
                split_of_group[group_id] = "validation"
            else:
                split_of_group[group_id] = "test"

    for row in rows:
        row["split"] = split_of_group[row["group_id"]]
    return rows


def check_dataset(rows, retrieval_eval_questions=()):
    """Stop with a clear error if the dataset has problems."""
    seen = set()
    for row in rows:
        key = " ".join(row["question"].lower().split())
        if key in seen:
            raise ValueError("Duplicate question: " + row["question"])
        seen.add(key)
    for q in retrieval_eval_questions:
        if " ".join(q.lower().split()) in seen:
            raise ValueError("Question is also in retrieval_eval.csv (leakage): " + q)
    split_of_group = {}
    for row in rows:
        if split_of_group.setdefault(row["group_id"], row["split"]) != row["split"]:
            raise ValueError("A group is spread over two splits: " + str(row["group_id"]))


def save_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "label", "group_id", "category", "split"])
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["label"] = int(row["label"])
    return rows