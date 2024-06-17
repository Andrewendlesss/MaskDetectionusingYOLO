## Introduction:
The following document presents a comprehensive overview of a project focused on the detection of individuals not wearing masks. This project aims to address the critical public health need precipitated by the COVID-19 pandemic.

## Topic's Reason and Inspiration:
The COVID-19 pandemic has had a profound global impact, necessitating the adoption of various preventive measures, including the widespread use of face masks. Despite the development of numerous machines designed to detect individuals not wearing masks, the accuracy of these systems has been inadequate. This project was conceived to develop a highly accurate model capable of detecting maskless individuals effectively. Additionally, the project aims to support government and public health organizations by providing essential data. By accurately counting and reporting the number of individuals not wearing masks, the project seeks to contribute significantly to public health and safety efforts.

## Objective and Function:
The primary objective of this project is to enhance public health by identifying individuals who are not wearing masks. In public spaces, such as art museums, this model can quickly and accurately monitor mask compliance, thereby helping to prevent the spread of COVID-19. When individuals are found not wearing masks, different strategies are implemented based on the size of the area. In smaller areas, staff members can directly remind individuals to wear a mask. In larger areas, public notifications or announcements are used to ensure compliance.

## Implementation:
The implementation process involved utilizing a mask dataset containing 150 images sourced from Roboflow, a platform providing public datasets. This dataset was imported into the working environment using Google Colab. To facilitate the training of the model, modifications were made to the YAML file to specify the labels (0 - mask, 1 - no mask) and the total number of classes (2 classes). The model was trained using the existing YOLOv8n.pt file and the customized YAML file.

To verify the model's type and name, the print function was employed.

In the YOLO.py file, real-time detection capabilities were implemented using the notebook’s camera. This enables the model to process each frame and detect mask compliance in real-time.

The project comprises two services:

### Real-Time Camera Service: 
This service detects individuals without masks using a live camera feed. When a maskless individual is detected, a rectangular box with the text "Please put the mask on" appears around them. Additionally, a counter displayed at the top left of the screen tracks the number of individuals not wearing masks. The warning text is triggered only when maskless individuals are detected.

### Recorded Video Service: 
This service analyzes recorded video footage to detect individuals not wearing masks, providing a useful tool for post-event analysis and compliance monitoring.
The development and deployment of these services aim to provide a robust solution for monitoring and enforcing mask compliance in various public settings, thereby contributing to the mitigation of COVID-19 transmission and enhancing public health safety measures.






