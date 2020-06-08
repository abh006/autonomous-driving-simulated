# Project X

This is a modified version of <a href="https://github.com/naokishibuya/car-behavioral-cloning">naokishibuya/car-behavioral-cloning</a>

Clone the Udacity simulator from <a href="https://github.com/udacity/self-driving-car-sim">udacity/self-driving-car-sim</a>

## Steps for setting up the environment:
- Setup a virtual environment with python 3.8
- Install all the dependencies from req.txt

## Training the model
The CNN model we use is the modified version of the model described in a paper published by Nvidia. You can find the paper <a href="https://arxiv.org/pdf/1604.07316v1.pdf">here</a>.
This repository contains a pre-trained model but it may cause limited accuracy, since it was trained with a smaller dataset.

- Run the Udacity simulator in training mode and record your game play to a folder
- This will generate a folder named `IMG` and a CSV file `driving_log.csv`. The `IMG` folder contains all the frames captured from the game play, from cameras placed in 3 different angles ( left, center and right).
- The CSV file `driving_log.csv` contains a mapping of the set of images captured from a single moment and the corresponding steering angle, throttle, speed and brake values
- The CSV structure will be 
    <table border="1">
    <thead>
        <td>Center</td>
        <td>Left</td>
        <td>Right</td>
        <td>Steering angle</td>
        <td>Throttle</td>
        <td>Speed</td>
        <td>Brake</td>
    </thead>
    </table>
- Copy the CSV file and the contents of `IMG` folder into a single folder
- For training the model, run
    `
    python model.py -d path-of-the-folder-with-training-images-and-csv
    `
- This will print a summary of the model and start training the model.


## Running the server with the trained model
For running the server without recording the simulator output :
```
python drive.py name-of-model-file.h5 
```

For recording the simulator output:
```
python drive.py name-of-model-file.h5 run1
```
where run1 is the name of folder to which the recorded frames are to be stored

You can constuct a video from these frames by:
```
python video.py run1 --fps 30
```
This will combine all the frames in the folder run1 in to a video of 30fps. Default fps value is 60.