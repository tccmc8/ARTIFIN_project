from schedule import student_habits_train_and_batch_predict

if __name__ == "__main__":
    student_habits_train_and_batch_predict.serve(
        name="student-habits-monthly",
        cron="0 9 1 * *",
        parameters={"n_samples": 100},
    )