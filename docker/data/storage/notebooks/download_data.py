import os
import datasets
import datetime as dt

datasets.logging.set_verbosity_error()

# set your number of processors, essential for downloading data efficiently
NUM_PROC = 8

# names of all dataset categories
names = [
    'All_Beauty',
    'Toys_and_Games',
    'Cell_Phones_and_Accessories',
    'Industrial_and_Scientific',
    'Gift_Cards',
    'Musical_Instruments',
    'Electronics',
    'Handmade_Products',
    'Arts_Crafts_and_Sewing',
    'Baby_Products',
    'Health_and_Household',
    'Office_Products',
    'Digital_Music',
    'Grocery_and_Gourmet_Food',
    'Sports_and_Outdoors',
    'Home_and_Kitchen',
    'Subscription_Boxes',
    'Tools_and_Home_Improvement',
    'Pet_Supplies',
    'Video_Games',
    'Kindle_Store',
    'Clothing_Shoes_and_Jewelry',
    'Patio_Lawn_and_Garden',
    'Unknown',
    'Books',
    'Automotive',
    'CDs_and_Vinyl',
    'Beauty_and_Personal_Care',
    'Amazon_Fashion',
    'Magazine_Subscriptions',
    'Software',
    'Health_and_Personal_Care',
    'Appliances',
    'Movies_and_TV'
]

# only want to look at data in 2023 - get Jan 1 2023 and convert to millis since epoch
start_time_millis = dt.datetime(2023, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc).timestamp() * 1_000

# download relevant portions of each dataset and write to local parquet
for name in names:

    # download and filter review data if it does not exist
    if not os.path.exists(f"amazon-data/reviews/{name}.parquet"):
        print(f"Writing {name} review data...")
        
        # load review data
        review_dataset = datasets.load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_review_{name}",
            num_proc=NUM_PROC,
            trust_remote_code=True)['full']
        
        # select columns of interest and filter for post-2023 before writing
        filtered_review_dataset = (
            review_dataset
            .select_columns(["rating", "title", "text", "parent_asin", "user_id", "timestamp"])
            .filter(lambda timestamp: timestamp >= start_time_millis, input_columns="timestamp")
        )
        filtered_review_dataset.to_parquet(f"amazon-data/reviews/{name}.parquet")

    if not os.path.exists(f"amazon-data/items/{name}.parquet"):
        print(f"Writing {name} item data...")
        
        # load item metadata
        meta_dataset = datasets.load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_meta_{name}",
            num_proc=NUM_PROC,
            trust_remote_code=True)['full']
        
        # select columns of interest before writing
        filtered_meta_dataset = (
            meta_dataset
            .select_columns(["main_category", "title", "average_rating", "rating_number", "parent_asin"])
        )
        filtered_meta_dataset.to_parquet(f"amazon-data/items/{name}.parquet")