import os
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import PhotoImage
from PIL import Image, ImageTk
from shutil import copy2
import flash
from flash.core.data.utils import download_data
from flash.image import SemanticSegmentation, SemanticSegmentationData

import matplotlib.pyplot as plt
import numpy as np
import torch

# Fonction pour déposer une image
def deposit_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Chargement de l'image sélectionnée
        image = Image.open(file_path)
        max_size = (255, 255)
        image.thumbnail(max_size)
        image2 = ImageTk.PhotoImage(image)
        image_label.config(image=image2)
        image_label.image = image2
        # Demande à l'utilisateur s'il souhaite enregistrer l'image
        answer = messagebox.askyesno("Enregistrer l'image", "Voulez-vous enregistrer l'image?")
        if answer:
            # Création du dossier "imageEntre" s'il n'existe pas déjà
            if not os.path.exists("imageEntre"):
                os.makedirs("imageEntre")
            # Détermination du prochain numéro d'image
            next_number = 1
            while os.path.exists(f"imageEntre/{next_number}.png"):
                next_number += 1
            # Enregistrement de l'image déposée dans le dossier "imageEntre" avec le nom correspondant
            image.save(f"imageEntre/{next_number}.png")
            # Affichage d'un message de confirmation
            messagebox.showinfo("Image déposée", "L'image a été déposée avec succès")



def process_image():
    if not os.path.exists("imageEntre"):
        messagebox.showinfo("Aucun dossier", "Aucun dossier")
    elif len(os.listdir("imageEntre")) == 0:
        messagebox.showinfo("Aucune image", "Aucune image")
    else:
        if not os.path.exists("imageSort"):
            os.makedirs("imageSort")
        #Si il y a des image dans le dossier imageSort
        if len(os.listdir("imageSort")) != 0:
            #On supprime tout les fichier dans le dossier imageSort
            for image in os.listdir("imageSort"):
                os.remove("imageSort/" + image)

        #model = SemanticSegmentation(
        #    backbone="mobilenetv3_large_100",
        #    head="fpn",
        #    num_classes=21,
        #)
        model = SemanticSegmentation.load_from_checkpoint('../model/' + model_choice.get())
        trainer = flash.Trainer(max_epochs=3, gpus=0)  # torch.cuda.device_count())
        applyModel(model,trainer)

def applyModel(model,trainer):
    #Segment all image
    allImage = []
    for image in os.listdir("imageEntre"):
        if image.endswith('.jpg') or image.endswith('.png'):
            allImage.append("imageEntre/" + image)
    datamodule = SemanticSegmentationData.from_files(
        predict_files=allImage,
        batch_size=1,
    )
    predictions = trainer.predict(model, datamodule=datamodule)
    for i in range(len(allImage)):
        #in_im_test = predictions[i][0]['input']
        out_im_test = predictions[i][0]['preds']

        #in_im_test = in_im_test.numpy().transpose(1, 2, 0)
        #in_im_test = (in_im_test - np.min(in_im_test)) / (np.max(in_im_test) - np.min(in_im_test))

        out_im_test = torch.argmax(out_im_test, 0)

        plt.imsave("imageSort/" + str(i+1) + ".png",out_im_test)
    #Affichage popup fin
    popup = tk.Toplevel(root)
    popup.grab_set()
    tk.Label(popup, text="Fini!").pack()


# Création de la fenêtre principale
root = tk.Tk()
root.geometry("400x400")
root.title("Dépôt d'image")
root.iconbitmap("icon.ico")

# Récupérer le nom de tout les modèle présent
listModel = []
for filename in os.listdir("../model/"):
    if filename.endswith(".pt"):
        listModel.append(filename)

#Menu déroulant pour le choix du modèle
model_choice = tk.StringVar(root)
model_choice.set(listModel[0])
model_choice_menu = tk.OptionMenu(root, model_choice, *listModel)
model_choice_menu.pack()

#Ajouter de l'espace
tk.Label(root, text="").pack()

# Texte de bienvenue
welcome_text = tk.Label(root, text="Bienvenue, veuillez déposer des images")
#Rendre le texte plus gros
welcome_text.config(font=("Courier", 13, "bold"))
welcome_text.pack()

# Affiche de l'image
image_label = tk.Label(root, text="Aucune image sélectionnée")
image_label.pack()

# Bouton pour déposer une image
btn_deposit = tk.Button(root, text="Déposer une image", command=deposit_image)
btn_deposit.pack()

# Bouton pour "process l'image"
btn_process = tk.Button(root, text="Process la/les image(s)", command=process_image)
btn_process.pack()

# Lancement de la boucle tkinter
root.mainloop()
