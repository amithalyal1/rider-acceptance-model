import pandas as pd
from src.utils.store import AssignmentStore



def main():
    store = AssignmentStore()

    booking_df = store.get_raw("booking_log.csv")
    booking_df = clean_booking_df(booking_df)

    participant_df = store.get_raw("participant_log.csv")
    participant_df = clean_participant_df(participant_df)

    dataset = merge_dataset(booking_df, participant_df)

    store.put_processed("dataset.csv", dataset)


def clean_booking_df(df: pd.DataFrame) -> pd.DataFrame:
    unique_columns = [
        "order_id",
        "trip_distance",
        "pickup_latitude",
        "pickup_longitude",
    ]
    df = df.drop_duplicates(subset=unique_columns)
    return df[unique_columns]


def clean_participant_df(df: pd.DataFrame) -> pd.DataFrame:   
    df = df.drop_duplicates()
    return df


def merge_dataset(bookings: pd.DataFrame, participants: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(participants, bookings, on="order_id", how="left")
    df = df.dropna()

    return df


if __name__ == "__main__":
    main()
