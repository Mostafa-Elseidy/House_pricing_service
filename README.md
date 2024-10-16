# Predicting House Prices Service

--------------------------------------------

## Objective

training an ML regression model and deploy it as web service with FLask for predicting new house price
-------------------------------------------

## Dataset info

* Square_Feet: Numeric - The size of the house in square feet.
* Bedrooms: Numeric - The number of bedrooms in the house.
* Age: Numeric - The age of the house in years.
* Location_Rating: Numeric - A rating of the location (1 to 10).
* Price: Numeric - The price of the house (target variable).

## Steps

* EDA and ML modeling
  * Python and Jupiter notebook
* Dependency and environment management
  * Conda, pipenv
* Deployment
  * Flask

-----------------------------  

## Try it

1. Clone files
2. pip install the requirements
3. Run the train.py script

run the web app locally on Ubuntu

``` bash
> gunicorn --bind=0.0.0.0:9696 predict:app
```

## Environment

Ubuntu Linux
