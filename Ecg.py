from skimage.io import imread
from skimage import color
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu, gaussian
from skimage.transform import resize
from skimage import measure
from sklearn.preprocessing import MinMaxScaler
from natsort import natsorted
import pandas as pd
import numpy as np
import os
import joblib

class ECG:
    def __init__(self):
        self.process_folder = os.path.join(os.getcwd(), 'process')
        os.makedirs(self.process_folder, exist_ok=True)

    def getImage(self, image_path):
        return imread(image_path)

    def GrayImgae(self, image):
        image_gray = color.rgb2gray(image)
        image_gray = resize(image_gray, (1572, 2213))
        return image_gray

    def DividingLeads(self, image):
        Leads = [
            image[300:600, 150:643], image[300:600, 646:1135], image[300:600, 1140:1625], image[300:600, 1630:2125],
            image[600:900, 150:643], image[600:900, 646:1135], image[600:900, 1140:1625], image[600:900, 1630:2125],
            image[900:1200, 150:643], image[900:1200, 646:1135], image[900:1200, 1140:1625], image[900:1200, 1630:2125],
            image[1250:1480, 150:2125]
        ]

        fig, ax = plt.subplots(4, 3)
        fig.set_size_inches(10, 10)
        x_counter, y_counter = 0, 0

        for i, lead in enumerate(Leads[:-1]):
            ax[x_counter][y_counter].imshow(lead)
            ax[x_counter][y_counter].axis('off')
            ax[x_counter][y_counter].set_title(f"Lead {i+1}")
            if (i + 1) % 3 == 0:
                x_counter += 1
                y_counter = 0
            else:
                y_counter += 1

        fig.savefig(os.path.join(self.process_folder, 'Leads_1-12_figure.png'))

        fig1, ax1 = plt.subplots()
        fig1.set_size_inches(10, 10)
        ax1.imshow(Leads[-1])
        ax1.axis('off')
        ax1.set_title("Lead 13")
        fig1.savefig(os.path.join(self.process_folder, 'Long_Lead_13_figure.png'))

        return Leads

    def PreprocessingLeads(self, Leads):
        fig, ax = plt.subplots(4, 3)
        fig.set_size_inches(10, 10)
        x_counter, y_counter = 0, 0

        for i, lead in enumerate(Leads[:-1]):
            gray = color.rgb2gray(lead)
            blurred = gaussian(gray, sigma=1)
            threshold = threshold_otsu(blurred)
            binary = blurred < threshold
            binary = resize(binary, (300, 450))

            ax[x_counter][y_counter].imshow(binary, cmap='gray')
            ax[x_counter][y_counter].axis('off')
            ax[x_counter][y_counter].set_title(f"Preprocessed Lead {i+1}")

            if (i + 1) % 3 == 0:
                x_counter += 1
                y_counter = 0
            else:
                y_counter += 1

        fig.savefig(os.path.join(self.process_folder, 'Preprossed_Leads_1-12_figure.png'))

        lead13 = Leads[-1]
        gray = color.rgb2gray(lead13)
        blurred = gaussian(gray, sigma=1)
        threshold = threshold_otsu(blurred)
        binary = blurred < threshold

        fig2, ax2 = plt.subplots()
        fig2.set_size_inches(10, 10)
        ax2.imshow(binary, cmap='gray')
        ax2.axis('off')
        ax2.set_title("Preprocessed Lead 13")
        fig2.savefig(os.path.join(self.process_folder, 'Preprossed_Leads_13_figure.png'))

    def SignalExtraction_Scaling(self, Leads):
        fig, ax = plt.subplots(4, 3)
        x_counter, y_counter = 0, 0

        for i, lead in enumerate(Leads[:-1]):
            gray = color.rgb2gray(lead)
            blurred = gaussian(gray, sigma=0.7)
            threshold = threshold_otsu(blurred)
            binary = blurred < threshold
            binary = resize(binary, (300, 450))
            contours = measure.find_contours(binary, 0.8)
            longest = max(contours, key=lambda c: c.shape[0])
            resized = resize(longest, (255, 2))

            ax[x_counter][y_counter].invert_yaxis()
            ax[x_counter][y_counter].plot(resized[:, 1], resized[:, 0], linewidth=1, color='black')
            ax[x_counter][y_counter].axis('image')
            ax[x_counter][y_counter].set_title(f"Contour {i+1}")

            if (i + 1) % 3 == 0:
                x_counter += 1
                y_counter = 0
            else:
                y_counter += 1

            scaled = MinMaxScaler().fit_transform(resized)
            df = pd.DataFrame(scaled[:, 0]).T
            df.columns = [f'X{j}' for j in range(df.shape[1])]
            df.to_csv(os.path.join(self.process_folder, f'Scaled_1DLead_{i+1}.csv'), index=False)

        fig.savefig(os.path.join(self.process_folder, 'Contour_Leads_1-12_figure.png'))

    def CombineConvert1Dsignal(self):
        files = natsorted([f for f in os.listdir(self.process_folder) if f.startswith("Scaled_1DLead") and f.endswith(".csv")])
        dfs = [pd.read_csv(os.path.join(self.process_folder, f)) for f in files]
        combined = pd.concat(dfs, axis=1, ignore_index=True)
        return combined

    def DimensionalReduciton(self, df):
        pca = joblib.load('model/PCA_ECG.pkl')
        reduced = pca.transform(df)
        return pd.DataFrame(reduced)

    def ModelLoad_predict(self, df):
        model = joblib.load('model/Heart_Disease_Prediction_using_ECG.pkl')
        result = model.predict(df)
        if result[0] == 0:
            return "You ECG corresponds to Abnormal Heartbeat"
        elif result[0] == 1:
            return "You ECG corresponds to Myocardial Infarction"
        elif result[0] == 2:
            return "Your ECG is Normal"
        else:
            return "You ECG corresponds to History of Myocardial Infarction"
