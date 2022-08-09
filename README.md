# CAAR


Datasets: 

(1) Frappe, frappe.csv and meta.csv, user-item interactions and attributes of items respectively

(2) Yelp, yelp_academic_dataset_review.json and yelp_academic_dataset_business.json, user-item interactions and attributes of items respectively. Due to the limit of document size, we can not upload the whole datset where but ths dataset is available at https://www.yelp.com/dataset

-- In main.py, we start specify the hyperparameters and start the experiment

-- In pre_traitement.py, we process the data to extract contextual information and attributes of items

-- In split_data.py, we split the dataset into training, test, and validation set (8:1:1)

-- In model.py, we construct the CAAR model

-- In train.py, we initialize and train the model



To rune the code, python3 main.py --name = 'Frappe' (This is to run experiment on frappe dataset, to run experiment on Yelp, simplely change to python3 main.py --name = 'Yelp')
