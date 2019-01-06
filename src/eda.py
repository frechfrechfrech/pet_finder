import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('max_columns',500)


# Data field info from kaggle data page
# Data Fields
# PetID - Unique hash ID of pet profile
# AdoptionSpeed - Categorical speed of adoption. Lower is faster. This is the value to predict. See below section for more info.
# Type - Type of animal (1 = Dog, 2 = Cat)
# Name - Name of pet (Empty if not named)
# Age - Age of pet when listed, in months
# Breed1 - Primary breed of pet (Refer to BreedLabels dictionary)
# Breed2 - Secondary breed of pet, if pet is of mixed breed (Refer to BreedLabels dictionary)
# Gender - Gender of pet (1 = Male, 2 = Female, 3 = Mixed, if profile represents group of pets)
# Color1 - Color 1 of pet (Refer to ColorLabels dictionary)
# Color2 - Color 2 of pet (Refer to ColorLabels dictionary)
# Color3 - Color 3 of pet (Refer to ColorLabels dictionary)
# MaturitySize - Size at maturity (1 = Small, 2 = Medium, 3 = Large, 4 = Extra Large, 0 = Not Specified)
# FurLength - Fur length (1 = Short, 2 = Medium, 3 = Long, 0 = Not Specified)
# Vaccinated - Pet has been vaccinated (1 = Yes, 2 = No, 3 = Not Sure)
# Dewormed - Pet has been dewormed (1 = Yes, 2 = No, 3 = Not Sure)
# Sterilized - Pet has been spayed / neutered (1 = Yes, 2 = No, 3 = Not Sure)
# Health - Health Condition (1 = Healthy, 2 = Minor Injury, 3 = Serious Injury, 0 = Not Specified)
# Quantity - Number of pets represented in profile
# Fee - Adoption fee (0 = Free)
# State - State location in Malaysia (Refer to StateLabels dictionary)
# RescuerID - Unique hash ID of rescuer
# VideoAmt - Total uploaded videos for this pet
# PhotoAmt - Total uploaded photos for this pet
# Description - Profile write-up for this pet. The primary language used is English, with some in Malay or Chinese.

# Import the training data
train = pd.read_csv('data/train.csv')
print('\n------------------------------\nSummary Stats of Train Data\n------------------------------\n')
print(train.describe())
print('\n-----------------------------\nData types of Train Data\n-----------------------------\n')
print(train.info())

# Plot distribution of the target = AdoptionSpeed, video amount, photo amount, and age
train[['AdoptionSpeed','VideoAmt','PhotoAmt','Age','MaturitySize']].hist()
plt.show()


# How often are breed and color nonzero?
breed_color = train.loc[:,(train.columns.str.contains('Breed'))| (train.columns.str.contains('Color'))]
proportion_populated = ((breed_color!=0).sum())/breed_color.shape[0]
print('\n------------------------------------------------------------\n\
Proportion Of Rows Populated for Breed and Color Fields\n---------------\
---------------------------------------------\n')
print(proportion_populated.round(2))
