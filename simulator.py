import uuid
import random
import string
import time
import json
import pandas as pd


class Simulator:
    """Simulation for a wearable sensor"""

    def __init__(self):
        """Initialized initial arguments"""
        self.current_timestamp = int(time.time())
        self.user_id = "".join(random.choices(string.ascii_lowercase, k=3))
        self.current_df = pd.DataFrame()

    def process_dataframe(self, user, second_count, current_df):
        """Processor function to create pandas dataframe from dict"""
        new_df = pd.DataFrame(user, index=[second_count])
        current_df = pd.concat([current_df, new_df])
        return current_df

    def write_to_json_file(self, users):
        """Return an output json file with result records updated"""
        with open("simulator_data.json", "w") as outfile:
            simulator_data = {"user_data": users}
            user_object = json.dumps(simulator_data, indent=4)
            outfile.write(user_object)

    def compute_for_quater(self, current_df):
        """Avg heart rate, min heart rate, max heart rate, avg resp rate, start seg,
			end seg computation for quater of an hour(15 mins)
		"""
        start_time_counter = 0
        end_time_counter = 15 * 60
        report_df = pd.DataFrame()
        while end_time_counter <= 2 * 60 * 60:
            heart_rate_matrix = (
                current_df[start_time_counter:end_time_counter]
                .groupby("user_id")["heart_rate"]
                .agg(min_hr=pd.np.min, max_hr=pd.np.max, avg_hr=pd.np.mean)
            )

            respiratory_rate_matrix = (
                current_df[start_time_counter:end_time_counter]
                .groupby("user_id")["respiratory_rate"]
                .agg(avg_rr=pd.np.mean)
            )

            timestamp_matrix = (
                current_df[start_time_counter:end_time_counter]
                .groupby("user_id")["timestamp"]
                .agg(start_seg=pd.np.min, end_seg=pd.np.max)
            )
            computation_df = pd.merge(
                respiratory_rate_matrix, heart_rate_matrix, on="user_id"
            )
            final_df = pd.merge(timestamp_matrix, computation_df, on=["user_id"])
            report_df = pd.concat([report_df, final_df])
            start_time_counter = end_time_counter
            end_time_counter = end_time_counter + 15 * 60
        report_df.to_csv("15_min_segment.csv")

    def generate_data(self):
        """generate data wearble sensor emits every second for the duration
			of two hours.
		"""
        
        users = []
        for second_count in range(1, 2 * 60 * 60 + 1):
            user = {}
            user["user_id"] = self.user_id
            user["heart_rate"] = random.randint(60, 101)
            user["timestamp"] = self.current_timestamp + second_count
            user["respiratory_rate"] = random.randint(12, 61)
            user["activity"] = random.randint(1, 10)
            self.current_df = self.process_dataframe(
                user, second_count, self.current_df
            )
            users.append(user)
        self.write_to_json_file(users)
        self.compute_for_quater(self.current_df)

    def compute_matrix(
        self, dataframe, start_time, end_time, groupbyattr, column_label, operator
    ):
        return (
            dataframe[start_time:end_time]
            .groupby(groupbyattr)[column_label]
            .agg(**{column_label: operator})
        )

    def compute_for_hour(self):
        """Avg heart rate, min heart rate, max heart rate, avg resp rate, start seg,
			end seg computation of an hour from segments of 15 mins available each
		"""
        minute_df = pd.read_csv("15_min_segment.csv")
        start_time = 0
        end_time = 4
        hour_report_df = pd.DataFrame()
        while end_time <= 8:
            avg_rr_matrix = self.compute_matrix(
                minute_df, start_time, end_time, "user_id", "avg_rr", pd.np.mean
            )
            avg_hr_matrix = self.compute_matrix(
                minute_df, start_time, end_time, "user_id", "avg_hr", pd.np.mean
            )
            min_hr_matrix = self.compute_matrix(
                minute_df, start_time, end_time, "user_id", "min_hr", pd.np.min
            )
            max_hr_matrix = self.compute_matrix(
                minute_df, start_time, end_time, "user_id", "max_hr", pd.np.max
            )
            start_timestamp_matrix = self.compute_matrix(
                minute_df, start_time, end_time, "user_id", "start_seg", pd.np.min
            )
            end_timestamp_matrix = self.compute_matrix(
                minute_df, start_time, end_time, "user_id", "end_seg", pd.np.max
            )
            avg_rr_hr_df = pd.merge(avg_hr_matrix, avg_rr_matrix, on="user_id")
            avg_min_rr_hr_df = pd.merge(avg_rr_hr_df, min_hr_matrix, on="user_id")
            avg_min_max_rr_hr_dataframe = pd.merge(
                avg_min_rr_hr_df, max_hr_matrix, on="user_id"
            )
            timestamp_df = pd.merge(
                start_timestamp_matrix, avg_min_max_rr_hr_dataframe, on="user_id"
            )
            final_timestamp_df = pd.merge(
                timestamp_df, end_timestamp_matrix, on="user_id"
            )
            hour_report_df = pd.concat([hour_report_df, final_timestamp_df])
            start_time = end_time
            end_time = end_time + 4
        hour_report_df.to_csv("avg_hour_segment.csv")


Simulator().generate_data()
Simulator().compute_for_hour()
