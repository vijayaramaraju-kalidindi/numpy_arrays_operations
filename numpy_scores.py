import numpy as np

# Task 1 — Generate and Inspect the Data

np.random.seed(42)
scores = np.random.randint(50, 101, (5,4))
print(f"Scores:\n{scores}")

third_student_second_subject_score = scores[2, 1]
print(f"The score of the 3rd student in the 2nd subject: {third_student_second_subject_score}")

last_two_students_scores = scores[3:, :]
print(f"All scores of the last 2 students (all subjects): \n {last_two_students_scores}")

first_three_students_first_two_subjects_scores = scores[:3, 1:3]
print(f"All scores for the first 3 students in subjects 2 and 3 only: \n {first_three_students_first_two_subjects_scores}")

# Task 2 — Analyze with Broadcasting 

mean_scores = np.mean(scores, axis=0)
print(f"column-wise mean: {mean_scores.round(2)}")

curve = np.array([5,3,7,2])
curved_scores = scores + curve
print(f"curved scores:\n{curved_scores}")

row_wise_max = np.max(curved_scores, axis=1)
print(f"row_wise_max: {row_wise_max}")

# Task 3 — Normalize and Identify

row_min = np.min(curved_scores, axis=1, keepdims=True)
row_max = np.max(curved_scores, axis=1, keepdims=True)

normalized_scores = (curved_scores - row_min) / (row_max - row_min)
print(f"normalized scores:\n{normalized_scores}")

max_index = np.unravel_index(np.argmax(normalized_scores), normalized_scores.shape)
#print(f"max_index: {max_index}")

print(f"student index (row): {max_index[0]}")
print(f"subject index (column): {max_index[1]}")

above_90 = curved_scores[curved_scores > 90]
print(f"curved scores above 90: {above_90}")

