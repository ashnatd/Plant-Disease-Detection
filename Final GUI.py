#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
from PIL import ImageTk, Image as PILImage
import cv2
import numpy as np
import os
import tensorflow as tf
import requests
from bs4 import BeautifulSoup
from io import BytesIO
import re
import textwrap

camera_index = 0  # Index of the camera to be used (default: 0)
output_image_path = "captured_leaf.jpg"  # Path to store the captured image
captured = False  # Flag to track if an image has been captured

def load_model():
    return tf.keras.models.load_model("trained_model_trying.h5")

def preprocess_image(image):
    try:
        if isinstance(image, str):  # If input is a file path
            image = PILImage.open(image)
        elif isinstance(image, np.ndarray):  # If input is already a NumPy array
            image = PILImage.fromarray(image)
        else:
            raise ValueError("Invalid input type for image.")
        
        image = image.resize((128, 128))  # Resize the image to match the model input size
        input_arr = np.array(image)
        input_arr = np.expand_dims(input_arr, axis=0)
        return input_arr
    except Exception as e:
        messagebox.showerror("Error", f"Failed to preprocess image: {str(e)}")
        return None  # Return None in case of error

def model_prediction(model, input_arr):
    try:
        predictions = model.predict(input_arr)
        class_index = np.argmax(predictions)
        return class_index
    except Exception as e:
        messagebox.showerror("Error", f"Failed to make predictions: {str(e)}")
        return None  # Return None in case of error

def fetch_disease_information(class_names):
    try:
        if class_names=="Orange__Haunglongbing(Citrus_greening)":
            url=f"https://sites.google.com/view/leafdiseases/home/orange/haunglongbingcitrus_greening"
            response=requests.get(url)
            print(url)
        if class_names=="Grape__Leaf_blight(Isariopsis_Leaf_Spot)":
            url=f"https://sites.google.com/view/leafdiseases/home/grape/leaf_blightisariopsis_leaf_spot"
            response=requests.get(url)
            print(url)
        if class_names=="Grape__Esca(Black_Measles)":
            url=f"https://sites.google.com/view/leafdiseases/home/grape/escablack_measles"
            response=requests.get(url)
            print(url)
        else:
            print(class_names)
            class_names=class_names.lower()
            new_class_name = class_names.replace(" ", "_")
            print(new_class_name)
            x = new_class_name.split('___')
            print(x)
            if(len(x)==1):
                added_underscore = new_class_name.replace('_','___',1)
                x = added_underscore.split('___')
            newurl = f"https://sites.google.com/view/leafdiseases/home/{x[0]}/{x[1]}"
            print(newurl)
            response = requests.get(newurl)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")

            headings = soup.find_all('span', class_='C9DxTc')

            images = soup.find_all('img', class_='CENy8b')

            info_window = tk.Toplevel(root)
            info_window.title("Disease Information")
            info_window.geometry("800x600") 

            info_frame = tk.Frame(info_window)
            info_frame.pack(fill=tk.BOTH, expand=True)

            scrollbar = tk.Scrollbar(info_frame, orient=tk.VERTICAL)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            info_canvas = tk.Canvas(info_frame, yscrollcommand=scrollbar.set)
            info_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            scrollbar.config(command=info_canvas.yview)

            content_frame = tk.Frame(info_canvas)
            info_canvas.create_window((0, 0), window=content_frame, anchor=tk.NW)

            for heading in headings:
                
                heading_label = tk.Label(content_frame, text=heading.get_text(), font=("Arial", 14, "bold"))
                heading_label.pack(anchor=tk.W)

                next_sibling = heading.find_next_sibling()

                if next_sibling and (next_sibling.name == 'p' or next_sibling.name == 'h3'):
                    
                    text_label = tk.Label(content_frame, text=next_sibling.text.strip(), font=("Arial", 12))
                    text_label.pack(anchor=tk.W)

            for image in images:
                img_url = image['src']
                response = requests.get(img_url)
                if response.status_code == 200:
                    img_data = response.content
                    
                    img = PILImage.open(BytesIO(img_data))
                    photo = ImageTk.PhotoImage(img)
                    img_label = tk.Label(content_frame, image=photo)
                    img_label.image = photo
                    img_label.pack(anchor=tk.W)
                else:
                    print("Failed to retrieve image:", img_url)

            content_frame.update_idletasks()
            info_canvas.config(scrollregion=info_canvas.bbox("all"))
            
        else:
            messagebox.showerror("Error", "Failed to fetch data. Status code: " + str(response.status_code))
    except Exception as e:
        messagebox.showerror("Error", "Failed to fetch data. Exception: " + str(e))
        print("Error:", e)


def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        input_image_path.set(file_path)
        display_image(file_path)

def display_image(image_path):
    try:
        image = PILImage.open(image_path)
        image = image.resize((300, 300))
        photo = ImageTk.PhotoImage(image)
        canvas.image = photo
        canvas.create_image(355, 60, anchor=tk.NW, image=photo)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to display image: {str(e)}")

def capture_image():
    global captured
    captured = True
    cap = cv2.VideoCapture(camera_index)
    ret, frame = cap.read()
    cap.release() 
    if ret:
        cv2.imwrite(output_image_path, frame)
        display_image(output_image_path)

def retake_image():
    global captured
    if os.path.exists(output_image_path):
        os.remove(output_image_path)
    captured = False
    canvas.delete("all")
    clear_prediction_table()

def predict():
    global captured
    image_path = input_image_path.get()
    if not captured and not image_path:
        messagebox.showerror("Error", "Please capture an image or select an image.")
        return
    model = load_model()
    if model:
        if captured:
            image = cv2.imread(output_image_path)
        else:
            image = image_path
        input_arr = preprocess_image(image)
        if input_arr is not None:
            class_index = model_prediction(model, input_arr)
            if class_index is not None:
                class_name = class_names[class_index]
                update_prediction_table(class_name)
                fetch_disease_information(class_name)
            else:
                messagebox.showerror("Error", "Failed to predict class index.")
        else:
            messagebox.showerror("Error", "Failed to preprocess image.")

def update_prediction_table(class_name):
    prediction_table.insert('', 'end', values=(class_name, diseases[class_name]))

def clear_prediction_table():
    prediction_table.delete(*prediction_table.get_children())

# Main Tkinter application
root = tk.Tk()
root.title("Plant Disease Recognition")
root.attributes('-fullscreen', True)  # Set the application to full-screen

input_image_path = tk.StringVar()
label = tk.Label(root, text="Select an image:", font=("Arial", 18))
label.pack()

entry = tk.Entry(root, textvariable=input_image_path, width=50, font=("Arial", 14))
entry.pack()

browse_button = tk.Button(root, text="Browse", command=select_image, font=("Arial", 14))
browse_button.pack()

camera_button = tk.Button(root, text="Capture from Camera", command=capture_image, font=("Arial", 14))
camera_button.pack()

retake_button = tk.Button(root, text="Retake Image", command=retake_image, font=("Arial", 14))
retake_button.pack()

canvas = tk.Canvas(root, width=1000, height=600)
canvas.pack()

prediction_table_frame = tk.Frame(root)
prediction_table_frame.pack(pady=2)
prediction_table_columns = ('Name of Species and Disease','Disease Description')
prediction_table = ttk.Treeview(prediction_table_frame, columns=prediction_table_columns, show='headings')
prediction_table.heading('Name of Species and Disease', text='Species and Disease')
prediction_table.heading('Disease Description', text='Description')
prediction_table.column('#2', width=500, minwidth=300, stretch=tk.YES)
#prediction_table.tag_configure('wrap', wraplength=500)
prediction_table.pack()

predict_button = tk.Button(root, text="Predict", command=predict, font=("Arial", 16))
predict_button.pack()
predict_button.place(relx=0.5, rely=0.75, anchor=tk.CENTER)

# Load class names and diseases
class_names= ['Anthracnose Mango',
 'Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Bacterial Canker Mango',
 'Cherry___Powdery_mildew',
 'Cherry___healthy',
 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn___Common_rust',
 'Corn___Northern_Leaf_Blight',
 'Corn___healthy',
 'Cutting Weevil Mango',
 'Die Back Mango',
 'Gall Midge Mango',
 'Grape___Black_rot',
 'Grape__Esca(Black_Measles)',
 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Healthy Mango',
 'Not_A_Leaf',
 'Orange__Haunglongbing(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,bell__Bacterial_spot',
 'Pepper,bell__healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Powdery Mildew Mango',
 'Rose_Black Spot',
 'Rose_Downy Mildew',
 'Rose_Fresh Leaf',
 'Sooty Mould Mango',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']
diseases = {
    'Anthracnose Mango': 'Anthracnose is a common disease of mango caused by the fungus Colletotrichum gloeosporioides.',
    'Apple___Apple_scab': 'Apple scab is a common fungal disease of apple trees caused by the fungus Venturia inaequalis.',
    'Apple___Black_rot': 'Black rot is a bacterial disease of apple trees caused by the bacterium Pectobacterium carotovorum subsp. carotovorum.',
    'Apple___Cedar_apple_rust': 'Cedar apple rust is a fungal disease of apple trees caused by the fungus Gymnosporangium juniperi-virginianae.',
    'Apple___healthy': 'The apple tree is healthy.',
    'Bacterial Canker Mango': 'Bacterial canker is a bacterial disease of mango caused by the bacterium Xanthomonas campestris pv. mangiferaeindicae.',
    'Cherry___Powdery_mildew': 'Powdery mildew is a fungal disease of cherries caused by the fungus Podosphaera clandestina.',
    'Cherry___healthy': 'The cherry tree is healthy.',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot': 'Cercospora leaf spot and Gray leaf spot are fungal diseases of corn caused by the fungi Cercospora zeae-maydis and Cercospora zeina, respectively.',
    'Corn___Common_rust': 'Common rust is a fungal disease of corn caused by the fungus Puccinia sorghi.',
    'Corn___Northern_Leaf_Blight': 'Northern leaf blight is a fungal disease of corn caused by the fungus Exserohilum turcicum.',
    'Corn___healthy': 'The corn plant is healthy.',
    'Cutting Weevil Mango': 'Cutting weevil is a pest of mango caused by the insect Sternochetus mangiferae.',
    'Die Back Mango': 'Die back is a fungal disease of mango caused by the fungus Lasiodiplodia theobromae.',
    'Gall Midge Mango': 'Gall midge is a pest of mango caused by the insect Procontarinia mangiferae.',
    'Grape___Black_rot': 'Black rot is a fungal disease of grapes caused by the fungus Guignardia bidwellii.',
    'Grape__Esca(Black_Measles)': 'Esca, also known as Black Measles, is a fungal disease of grapes caused by the fungi Phaeomoniella chlamydospora and Phaeoacremonium minimum.',
    'Grape__Leaf_blight(Isariopsis_Leaf_Spot)': 'Leaf blight, also known as Isariopsis Leaf Spot, is a fungal disease of grapes caused by the fungus Isariopsis Leaf Spot.',
    'Grape___healthy': 'The grape vine is healthy.',
    'Healthy Mango': 'The mango fruit is healthy.',
    'Not_A_Leaf': 'This is not a leaf.',
    'Orange__Haunglongbing(Citrus_greening)': 'Huanglongbing, also known as Citrus Greening, is a bacterial disease of citrus trees caused by the bacterium Candidatus Liberibacter asiaticus.',
    'Peach___Bacterial_spot': 'Bacterial spot is a bacterial disease of peaches caused by the bacterium Xanthomonas campestris pv. pruni.',
    'Peach___healthy': 'The peach tree is healthy.',
'Pepper,bell__Bacterial_spot': 'Bacterial spot is a bacterial disease of bell peppers caused by the bacterium Xanthomonas campestris pv. vesicatoria.',
    'Pepper,bell__healthy': 'The bell pepper plant is healthy.',
    'Potato___Early_blight': 'Early blight is a fungal disease of potatoes caused by the fungus Alternaria solani.',
    'Potato___Late_blight': 'Late blight is a fungal disease of potatoes caused by the fungus Phytophthora infestans.',
    'Potato___healthy': 'The potato plant is healthy.',
    'Powdery Mildew Mango': 'Powdery mildew is a fungal disease of mango caused by the fungus Oidium mangiferae.',
    'Rose_Black Spot': 'Black spot is a fungal disease of roses caused by the fungus Diplocarpon rosae.',
    'Rose_Downy Mildew': 'Downy mildew is a fungal disease of roses caused by the fungus Peronospora sparsa.',
    'Rose_Fresh Leaf': 'This is a fresh leaf.',
    'Sooty Mould Mango': 'Sooty mould is a fungal disease of mango caused by various fungi in the family Capnodiales.',
    'Squash___Powdery_mildew': 'Powdery mildew is a fungal disease of squash caused by the fungus Podosphaera xanthii.',
    'Strawberry___Leaf_scorch': 'Leaf scorch is a fungal disease of strawberries caused by the fungus Diplocarpon earliana.',
    'Strawberry___healthy': 'The strawberry plant is healthy.',
    'Tomato___Bacterial_spot': 'Bacterial spot is a bacterial disease of tomatoes caused by the bacterium Xanthomonas campestris pv. vesicatoria.',
    'Tomato___Early_blight': 'Early blight is a fungal disease of tomatoes caused by the fungus Alternaria solani.',
    'Tomato___Late_blight': 'Late blight is a fungal disease of tomatoes caused by the fungus Phytophthora infestans.',
    'Tomato___Leaf_Mold': 'Leaf mold is a fungal disease of tomatoes caused by the fungus Fulvia fulva.',
    'Tomato___Septoria_leaf_spot': 'Septoria leaf spot is a fungal disease of tomatoes caused by the fungus Septoria lycopersici.',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Spider mites, also known as Two-spotted spider mites, are pests of tomatoes caused by the mite Tetranychus urticae.',
    'Tomato___Target_Spot': 'Target spot is a fungal disease of tomatoes caused by the fungus Corynespora cassiicola.',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Tomato yellow leaf curl virus is a viral disease of tomatoes caused by the virus Tomato yellow leaf curl virus.',
    'Tomato___Tomato_mosaic_virus': 'Tomato mosaic virus is a viral disease of tomatoes caused by the virus Tomato mosaic virus.',
    'Tomato___healthy': 'The tomato plant is healthy.'
}
root.mainloop()


# In[ ]:




