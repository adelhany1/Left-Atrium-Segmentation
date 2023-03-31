# Left-Atrium-Segmentation

The heart has four chambers: two atria and two ventricles.

The right atrium receives oxygen-poor blood from the body and pumps it to the right ventricle.
The right ventricle pumps the oxygen-poor blood to the lungs.
The left atrium receives oxygen-rich blood from the lungs and pumps it to the left ventricle.
The left ventricle pumps the oxygen-rich blood to the body.

![alt text](https://sa1s3optim.patientpop.com/assets/docs/346147.jpg)

Left atrial enlargement is when one of your heart chambers gets bigger than normal. This happens over time when your left atrium tries to adjust to issues such as high blood pressure in the rest of your heart.
Left atrial enlargement is a warning sign that one of your heart’s upper chambers is handling high pressure and too much blood. People with this issue often have high blood pressure, heart valve problems or other heart issues. Treatment varies depending on the cause. You may need medication, healthier habits or valve repair/replacement.

Your left atrium can get larger and stretch when it tries to adapt to make up for this high pressure and/or high volume. This stretching causes scarring and injury to your atrium. It’s like a big brother who tries to help his siblings carry the load but ends up getting hurt himself.

An Italian study of adults found that 12% of them developed left atrial enlargement during a period of 10 years. Based on this, researchers believe the condition isn’t rare in the general population. In the study, most of the people who developed left atrial enlargement were in their 40s and 50s.

Segmentation of the left atrium (LA) is required to evaluate atrial size and function, which are important imaging biomarkers for a wide range of cardiovascular conditions, such as atrial fibrillation, stroke, and diastolic dysfunction.

(Left) Atrium Segmentation classifies each voxel in the MRI into either "Not Left Atrium" or " Left Atrium" Enables exact volume calculation
Changes in atrial volume are assoicated with cardiac disorders, such as atrial fibrillation or mitral valve stenosis (Narrowing of the mitral valve orifice, blocking blood flow).

This Task is from [Medical Segmentation Decathlon](http://medicaldecathlon.com/) , The goal is to Automatically segment the left atrium in a 3D cardiac MRI images.

### Data
- Medical Segmentation Decathlon (http://medicaldecathlon.com/)
- 20 MRI scans of the heart with corresponding Ground Truth mask (https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2)
- 4542 2D MRI and label slices

Using Deep learning and python we will solve the problem.
* pytorch (preferably with CUDA)
* pytorch lightning
* nibabel
* matplotlib
* numpy
* pydicom
* tqdm
* celluloid
* imgaug


