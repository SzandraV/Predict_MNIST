#Imports
#import os.path
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import joblib

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import cv2

from PIL import Image
import torchvision
import torch

import digit_utils as utils
import time

#Definitions
#@st.cache
def load_dataset():
    
    global labels, number_of_classes

    # LOAD Dataset - MNIST
    mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
    X = mnist["data"]
    y = mnist["target"].astype(np.uint8)
    train_N = 60000 # Choose a suitable N
    X_tr = X[:train_N]
    y_tr = y[:train_N]
    
    # Initialization
    labels = np.unique(y_tr)
    number_of_classes = np.max(y_tr)+1

    ## Use StandardScaler()
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_tr.astype(np.float32))

    return X_train_scaled, y_tr

def load_my_digits():
    my_digits = np.empty((1, 28*28))

    for i in range(0, 10):
        my_digit_path = './My_digits/set_merged/num{}.jpg'.format(i)
        my_digit = utils.load_and_prep_digit(my_digit_path)
        img_to_test = np.reshape(my_digit, (1, 28*28))
        my_digits = np.vstack([my_digits, img_to_test])

    my_digits = np.delete(my_digits, 0, axis=0)

    return my_digits

def intro():
    print("Hello this is ML Model Platform for Your Digit's prediction :)")

def main_program():
    st.set_page_config(layout = "wide")
    st.header("ML Model Platform for Your Digit's prediction")

    if 'model_started' not in st.session_state:
        st.session_state['model_started'] = False

    if 'current_model' not in st.session_state:
        st.session_state['current_model'] = None

    ### Inputs / Variables / Datasets

    functions = ['My digits', 'Your digit']

    #Sidebar definition
    page = st.sidebar.selectbox('Test the Model with:', functions)
    if st.sidebar.button('Close'):
        st.write("That's all folks :) Thanks for now.")
        st.write("Please close the window and terminate the StreamLit Server in you Terminal.")
        st.stop()
        #Please respekt: "Streamlit effectively spins up a server, and
        # we do not envision a use case to shut down the server
        # due to a UI interaction in the script."

    #   Predict My digits
    if page == 'My digits':
        #st.write(f"Current loaded model: {st.session_state['current_model']}")
        col1, col2 = st.columns([2,4])
        with col1:
            t0 = time.time()
            if st.button('Load model'):
                print('{} Model is loading...'.format(page))
                with st.spinner(text = 'Digit prediction Model is loading... \n\nPlease respect that it could be take some minutes :)'.format(page)):
                    loaded_model = joblib.load('./models/best_SVM_model.sav'.format(page))
                    st.session_state['model_started'] = True
                    st.session_state['current_model'] = page
                    X_train_scaled, y_tr= load_dataset()
                    #loaded_model.fit(X_train_scaled, y_tr)
                t1 = time.time()
                print('Model loaded and it tooks {:.1f}s.'.format(t1 - t0))
                st.success('Model loaded and it tooks {:.1f}s.'.format(t1 - t0))

            if not st.session_state['model_started'] is True or st.session_state['current_model'] != page:
                st.warning('Please load in your model')
            else:
                my_digit_test = load_my_digits()
                print('My digits are loaded'.format(page))
                st.success('My digits are loaded and ready for prediction :)')
                own_pred = loaded_model.predict(my_digit_test)
                st.write("Model's accuracy on my digits: {}".format(accuracy_score(labels, own_pred)))
                st.write("That's all folks :) Thanks for now.")

        with col2:
            if not st.session_state['model_started'] is True or st.session_state['current_model'] != page:
                st.warning('Do not forget to load the model! to see the Prediction')
            else:
                # Visualize my original digits
                plt.figure()
                fig, axis = plt.subplots(ncols = number_of_classes, figsize=(32,4))
                fig.suptitle('My digits by Classes\n\n', fontsize = 30)

                for i in range(number_of_classes):
                    axis[i].set_ylabel(labels[i], rotation=0, labelpad = 50)
                    axis[i].set_title('Class {}'.format(labels[i]), loc='center', fontsize = 20)
                    axis[i].imshow(my_digit_test[i].reshape(28, 28), interpolation='nearest', aspect='auto')

                plt.tight_layout()
                st.pyplot(fig)

                # Visualize my digits predicted Classes
                plt.figure()
                fig, axis = plt.subplots(ncols = number_of_classes, figsize=(32,4))
                fig.suptitle("My digits's Predicted Class\n\n", fontsize = 30)

                for i in range(number_of_classes):
                    axis[i].set_ylabel(labels[i], rotation=0, labelpad = 50)
                    axis[i].set_title('Pred Class {}'.format(own_pred[i]), loc='center', fontsize = 20)
                    axis[i].imshow(my_digit_test[i].reshape(28, 28), interpolation='nearest', aspect='auto')

                plt.tight_layout()
                st.pyplot(fig)
    
    #   Predict your digits
    elif page == 'Your digit':

        # if st.button('Load model'):
        #     t0 = time.time()
        #     print('{} Model is loading...'.format(page))
        #     with st.spinner(text = 'Digit prediction Model is loading... \n\nPlease respect that it could be take some minutes :)'.format(page)):
        #         loaded_model = joblib.load('./models/best_SVM_model.sav'.format(page))
        #         st.session_state['model_started'] = True
        #         st.session_state['current_model'] = page
        #     t1 = time.time()
        #     print('Model loaded and it tooks {:.1f}s.'.format(t1 - t0))
        #     st.success('Model loaded and it tooks {:.1f}s.'.format(t1 - t0))
        
        # if not st.session_state['model_started'] is True or st.session_state['current_model'] != page:
        #     st.warning('Please load in your model')
        # else:
            realtime_update = st.sidebar.checkbox("Update in realtime", True)
            st.write('### Draw a digit in 0-9 in the box below')
            # Create a canvas component
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                stroke_width= 5,
                stroke_color='#FFFFFF',
                background_color='#000000',
                update_streamlit=realtime_update,
                height=200,
                width=200,
                drawing_mode='freedraw',
                key="canvas",
            )
            # Processing Cavas' input
            if canvas_result.image_data is not None:
                # Get the numpy array (4-channel RGBA 100,100,4)
                input_numpy_array = np.array(canvas_result.image_data)
                # Get the RGBA PIL image
                input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
                input_image.save('user_input.png')
                # Convert it to grayscale
                input_image_gs = input_image.convert('L')
                input_image_gs_np = np.asarray(input_image_gs.getdata()).reshape(200,200)
                # Create a temporary image for opencv to read it
                input_image_gs.save('temp_for_cv2.jpg')
                image = cv2.imread('temp_for_cv2.jpg', 0)
                
                # Centering
                # Start creating a bounding box
                height, width = image.shape
                x,y,w,h = cv2.boundingRect(image)
                # Create new blank image and shift ROI to new coordinates
                ROI = image[y:y+h, x:x+w]
                mask = np.zeros([ROI.shape[0]+10,ROI.shape[1]+10])
                width, height = mask.shape
                x = width//2 - ROI.shape[0]//2 
                y = height//2 - ROI.shape[1]//2 
                mask[y:y+h, x:x+w] = ROI
                # Check if centering/masking was successful
                output_image = Image.fromarray(mask) # mask has values in [0-255] as expected
                # Now we need to resize, but it causes problems with default arguments as it changes the range of pixel values to be negative or positive
                # compressed_output_image = output_image.resize((22,22))
                # Therefore, we use the following:
                compressed_output_image = output_image.resize((22,22), Image.BILINEAR) # PIL.Image.NEAREST or PIL.Image.BILINEAR also performs good
                convert_tensor = torchvision.transforms.ToTensor()
                tensor_image = convert_tensor(compressed_output_image)
                # Another problem we face is that in the above ToTensor() command, we should have gotten a normalized tensor with pixel values in [0,1]
                # But somehow it doesn't happen. Therefore, we need to normalize manually
                tensor_image = tensor_image/255.
                # Padding
                tensor_image = torch.nn.functional.pad(tensor_image, (3,3,3,3), "constant", 0)
                # Normalization shoudl be done after padding i guess
                convert_tensor = torchvision.transforms.Normalize((0.1307), (0.3081)) # Mean and std of MNIST
                tensor_image = convert_tensor(tensor_image)
                # st.write(tensor_image.shape) 
                # Shape of tensor image is (1,28,28)
                # The following gives noisy image because the values are from -1 to 1, which is not a proper image format
                im = Image.fromarray(tensor_image.detach().cpu().numpy().reshape(28,28), mode='L')
                im.save("processed_tensor.png", "PNG")
                # So we use matplotlib to save it instead
                plt.imsave('processed_tensor.png',tensor_image.detach().cpu().numpy().reshape(28,28), cmap='gray')
                
                if st.button('Predict'):
                    ### Compute the predictions
                    img_to_test = np.reshape(im, (1, 28*28))
                    loaded_model = joblib.load('./models/best_SVM_model.sav'.format(page))
                    own_pred = loaded_model.predict(img_to_test)
                    st.write('### Prediction') 
                    st.write('### '+str(own_pred[0]))

    else:
        st.warning('Please choose another model. This is under construction... o.O')
        st.session_state['current_model'] = None
        st.session_state['model_started'] = False


#Main area :)
if __name__ == "__main__":
    #Info to User
    intro()
    #Processing the Program :)
    main_program()
