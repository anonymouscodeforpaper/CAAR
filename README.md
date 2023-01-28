# CAAR
This is the implementation of our paper titled "CAAR: Context-aware argumentative explanations in recommender systems" submitted to CAISE 2023

## Main steps
There are mainly three steps:
- Computing the representation of target users under the target contextual situation;
- Computing users' ratings towards attributes of items under the given contextual situation;
- Aggregating the ratings in the previous step to get users' ratings towards items.
![Illustration of the framework of CAAR](https://github.com/anonymouscodeforpaper/CAAR/blob/main/framework.png)


## Datasets: 

(1) Frappe, frappe.csv and meta.csv, user-item interactions and attributes of items respectively

(2) Yelp, yelp_academic_dataset_review.json and yelp_academic_dataset_business.json, user-item interactions and attributes of items respectively. Due to the limit of document size, we can not upload the whole datset where but ths dataset is available at https://www.yelp.com/dataset

-- In main.py, we start specify the hyperparameters and start the experiment

-- In pre_traitement.py, we process the data to extract contextual information and attributes of items

-- In split_data.py, we split the dataset into training, test, and validation set (8:1:1)

-- In model.py, we construct the CAAR model

-- In train.py, we initialize and train the model



To run the code, python3 main.py --name = 'Frappe' (This is to run experiment on frappe dataset, to run experiment on Yelp, simplely change to python3 main.py --name = 'Yelp')
