from dataclasses import dataclass
import os
import sys
import constants
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models import ChatOpenAI

import numpy as np
import matplotlib.pyplot as plt
import mplcursors

CALORIE_GOAL_LIMIT = 2000 # kcal
PROTEIN_GOAL = 175 # grams
FAT_GOAL = 90 # grams
CARBS_GOAL = 205 # grams
os.environ["OPENAI_API_KEY"] = constants.APIKEY

if os.path.getsize("data.txt") == 0:
    with open("data.txt", "a") as file:
        file.write(f"""Goals: 
                   Calories: {CALORIE_GOAL_LIMIT} kcal, 
                   Protein:{PROTEIN_GOAL} grams, 
                   Fat: {FAT_GOAL} grams, 
                   Carbs: {CARBS_GOAL} grams\n""")
        


today = []
colors = ['coral', 'peachpuff', 'skyblue']
goal_colors = ['red', 'orange', 'deepskyblue']


@dataclass
class Food:
    name: str
    calories: int
    protein: int
    fat: int
    carbs: int


with open("data.txt", "r") as file:
    lines = file.readlines()
    if not lines:
        print("data.txt is empty.")
    else:
        for line in lines:
            if "Meal:" in line: # check if the line contains meal information
                meal_info = line.strip().split(", ")
                name = meal_info[0].split(": ")[1]
                calories_str = meal_info[1].split(": ")[1]
                calories = int(calories_str.split(" ")[0])
                proteins_str = meal_info[2].split(": ")[1]
                proteins = int(proteins_str.split(" ")[0])
                fats_str = meal_info[3].split(": ")[1]
                fats = int(fats_str.split(" ")[0])
                carbs_str = meal_info[4].split(": ")[1]
                carbs = int(carbs_str.split(" ")[0])
                food = Food(name, calories, proteins, fats, carbs)
                today.append(food)


    
done = False

while not done:
    print( """
    (1) Add a new food
    (2) Visualize progress
    (3) Ask a chatbot about your diet
    (q) Quit
    """)

    choice = input("Choose an option: ")

    if choice == "1":
        print("Adding a new food!")
        try:
            name = input("Name: ")
            calories = int(input("Calories: "))
            proteins = int(input("Protein: "))
            fats = int(input("Fat: "))
            carbs = int(input("Carbs: "))
            food = Food(name, calories, proteins, fats, carbs)
            today.append(food)
            print("Successfully added!")
            with open("data.txt", "a") as file:
                file.write(f"Meal: {name}, Calories: {calories} kcal, Protein: {proteins} grams, Fat: {fats} grams, Carbs: {carbs} grams\n")
        except ValueError:
            print("Error: Please enter valid integer values for calories, proteins, fats, and carbs.")
        
    elif choice == "2":
        if(today == []):
            print("There is nothing to visualize!")
        else:
            try:
                calorie_sum = sum(food.calories for food in today)
                protein_sum = sum(food.protein for food in today)
                fats_sum = sum(food.fat for food in today)
                carbs_sum = sum(food.carbs for food in today)

                fig, axs = plt.subplots(2,2)
                axs[0,0].pie([protein_sum, fats_sum, carbs_sum], labels=["Proteins", "Fats", "Carbs"], autopct="%1.1f%%",  colors=goal_colors)
                axs[0,0].set_title("Macronutrients Distribution")
                axs[0,1].bar([0,1,2], [protein_sum, fats_sum, carbs_sum], width =0.4, color=colors)
                axs[0,1].bar([0.5,1.5,2.5], [PROTEIN_GOAL, FAT_GOAL, CARBS_GOAL], width =0.4,color=goal_colors)
                axs[0,1].set_xticks([0.25, 1.25, 2.25])
                axs[0,1].set_xticklabels(["Protein", "Fats", "Carbs"])
                axs[0,1].legend(["Sum", "Goal"])
                axs[0,1].set_title("Macronutrients Progress")
                excess_calories = max(0, calorie_sum - CALORIE_GOAL_LIMIT)
                if excess_calories != 0:
                    axs[1,0].bar([0, 1], [calorie_sum, CALORIE_GOAL_LIMIT], width=0.4, color=["mediumseagreen", "springgreen"])
                    axs[1,0].set_xticks([0, 1])
                    axs[1,0].set_xticklabels(["Total Calories", "Calorie Goal"])
                    axs[1,0].set_title("Calorie Goal Met / Exceeded")
                else:
                    remaining_calories = max(0, CALORIE_GOAL_LIMIT - calorie_sum)
                    axs[1,0].pie([calorie_sum, remaining_calories], labels=["Calories", "Remaining"], autopct="%1.1f%%", colors=goal_colors)
                    axs[1,0].set_title("Calorie Goal Progress")
                food_names = [food.name for food in today]
                calories_consumed = [food.calories for food in today]
                axs[1, 1].bar(food_names, calories_consumed, color='mediumslateblue')
                axs[1, 1].set_title("Calories Consumed per Meal")
                axs[1, 1].set_xlabel("Meals")
                axs[1, 1].set_ylabel("Calories")
                axs[1, 1].tick_params(axis='x', rotation=45)
                fig.tight_layout()
                plt.show()
            except ValueError:
                print("Error: There was an issue graphing your progress.")

    elif choice =="3":
        query = query = input("What is your Question?: ")
        print(query)
        loader = TextLoader("data.txt")
        index = VectorstoreIndexCreator().from_loaders([loader])
        print(index.query(query, llm=ChatOpenAI()))
        
    elif choice == "q":
        clear = input("Would you like to clear your data? (y/n): ")
        if(clear == "y"):
            with open("data.txt", "w") as file:
                pass
            done= True
        else:
            done= True

    else:
        print("Invalid choice!")

