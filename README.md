# Furniture Classification

## How to use

1. Install all the requirements from requirements.txt

```bash
pip install -r requirements.txt
```

2. Upload your videos to the `.training_videos` folder

3. Run the `extract_frames.py` script to extract frames from the videos

```bash
python extract_frames.py
```

4. Label the frames in the `.training_frames/labels.csv` file

5. Run the `train.py` script to train the model as many times as needed

```bash
python train.py --learning_rate 0.0001 --epochs 2
```

6. Run the `classification_api.py` script to start the classification API

```bash
python classification_api.py
```

7. Use the classification API to classify your frames by sending them as binary body to the `POST http://127.0.0.1:4000/classify` endpoint

```bash
curl --request POST 'http://127.0.0.1:4000/classify' \
--header 'Content-Type: image/jpeg' \
--data-binary '@/path/to/your/frame.jpg'
```
