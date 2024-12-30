# import requests
# from PIL import Image
# from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

# url = "https://www.ilankelman.org/stopsigns/australia.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
# header_text1 = "Describe the content and function of the target widget, consider relations between other widgets and the target widget"

# model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-widget-captioning-base").to("cuda")
# processor = Pix2StructProcessor.from_pretrained("google/pix2struct-widget-captioning-base")


# # image only
# inputs = processor(images=image, return_tensors="pt", text=header_text1, is_vqa=True).to("cuda")

# predictions = model.generate(**inputs)
# print(processor.decode(predictions[0], skip_special_tokens=True))

from typing import List
from PIL import Image, ImageDraw
# import cv2
import requests
from transformers import AutoProcessor, Pix2StructForConditionalGeneration
import torch
from torch.nn import DataParallel
import tensorflow as tf
import os
from datetime import datetime
from numba import cuda

def set_tf():
# num_gpus=1
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    if gpus:
        try:
            # tf.config.experimental.set_visible_devices(gpus[6], 'GPU')
            tf.config.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True) #默认情况下，TensorFlow 会占用所有 GPU 的显存。可以通过以下方式限制显存占用
            tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
        )
            # Set TensorFlow to use GPU 0       
        except RuntimeError as e:
            print(e)


class widget_captioning:
    def __init__(self, gpus=None) -> None:
        # if gpus is not None:
        #     if isinstance(gpus, list):
        #         gpus1 = ",".join(map(str, gpus))
        #     os.environ["CUDA_VISIBLE_DEVICES"] = gpus1
        # else:
        #     os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
        # 初始化其他成员变量或执行其他初始化操作
        # print(f"CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}")
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
        # Check if CUDA is available and set the devices accordingly
        self.device_count = torch.cuda.device_count()
        self.devices = gpus  # 使用所有可用设备
        set_tf()
        print(self.devices)
        # self.device_ids_model1 = [0, 1, 2, 3, 4, 5, 6]  # GPUs for the first model
        # device_ids_model2 = [3, 4, 5]  # GPUs for the second model
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.cuda.empty_cache()
        # self.device = "cpu"
        self.model = Pix2StructForConditionalGeneration.from_pretrained("../dataset_Auto-UI/pix2struct-widget-captioning-large").to(self.devices[0])
        # self.model = DataParallel(self.model, device_ids=self.devices)  # Wrap the model with DataParallel
        self.processor = AutoProcessor.from_pretrained("../dataset_Auto-UI/pix2struct-widget-captioning-large")
                # Initialize a model and processor for each device

    # 检查是否有可用的 GPU
        # if torch.cuda.is_available():
        #     total_memory = torch.cuda.get_device_properties(0).total_memory
        #     reserved_memory = torch.cuda.memory_reserved(0)
        #     allocated_memory = torch.cuda.memory_allocated(0)
        #     free_memory = reserved_memory - allocated_memory
        #     available_memory = total_memory - reserved_memory

        #     print(f"总显存量：{total_memory / 1024 ** 2:.2f} MiB")
        #     print(f"已预留的显存量：{reserved_memory / 1024 ** 2:.2f} MiB")
        #     print(f"已分配的显存量：{allocated_memory / 1024 ** 2:.2f} MiB")
        #     print(f"当前可用的显存量：{free_memory / 1024 ** 2:.2f} MiB")
        #     print(f"总可用的显存量：{available_memory / 1024 ** 2:.2f} MiB")
        # else:
        #     print("没有可用的 GPU")
        
        # self.model_wc = Pix2StructForConditionalGeneration.from_pretrained("../dataset_Auto-UI/pix2struct-widget-captioning-large").to(self.devices[0])
        # to(self.device)
        # total_memory = torch.cuda.get_device_properties(0).total_memory
        # reserved_memory = torch.cuda.memory_reserved(0)
        # allocated_memory = torch.cuda.memory_allocated(0)
        # free_memory = reserved_memory - allocated_memory
        # available_memory = total_memory - reserved_memory

        # print(f"总显存量：{total_memory / 1024 ** 2:.2f} MiB")
        # print(f"已预留的显存量：{reserved_memory / 1024 ** 2:.2f} MiB")
        # print(f"已分配的显存量：{allocated_memory / 1024 ** 2:.2f} MiB")
        # print(f"当前可用的显存量：{free_memory / 1024 ** 2:.2f} MiB")
        # print(f"总可用的显存量：{available_memory / 1024 ** 2:.2f} MiB")
        
        # self.model_wc = DataParallel(self.model_wc, device_ids=self.device_ids_model1)  # Wrap the model with DataParallel
        # self.model_wc = DataParallel(self.model_wc)
        
    def mark_widget_for_wc(self, bbox: List[float], image, saveas_id):
        '''draw bounding box for target widget'''
        # (left, top, right, bottom)
        bbox = [int(coord) for coord in bbox]
        xmin, ymin, xmax, ymax = bbox
        # draw bounding box
        img_draw = ImageDraw.Draw(image, "RGBA")
        img_draw.rectangle(
            xy=((xmin, ymin),
                (xmax, ymax)),
            fill=(0, 0, 255, 0),
            outline=(0, 0, 255, 255))
        # image.save(f"dataset/images/image_with_bbox_{saveas_id}.png",format="PNG")
        return image
    
    def generate_widget_captioning(self, image, taskDesc: str, current_activity: str="not provided", device_id=0) -> str:
        model = self.model
        processor = self.processor
        device = self.devices[device_id]
        header_text1 = f"The Android UI is visited to finish the task ‘{taskDesc}’. The current activity for the UI page is ‘{current_activity}’. Describe the function of the target widget in UI with blue bounding box"
        print("header_text1: ",header_text1) # The Android UI page is visited to answer the question or finish the task Play the new Bruno Mars video on YouTube. The current activity for the UI page is activity for Play the new Bruno Mars video on YouTube. Describe the function of the target widget in UI page with blue bounding box
        # print("image: ",image)
        # import pickle
        # save image object
        # with safe_open(f"dataset/images/image_test.pkl", "wb") as f:
        #     pickle.dump(image, f)
        start_time = datetime.now()
        # cuda.select_device(0)
        # cuda.close()
        # tf.keras.backend.clear_session()
        inputs = processor(images=image, return_tensors="pt", text=header_text1, is_vqa=True).to(self.devices[0])
        end_time = datetime.now()
        print(f"Execution time1: {end_time - start_time}")
        # to(self.device)
        # print("inputs: ",inputs)
        
        # (self.device)
        # 
        # autoregressive generation
        start_time = datetime.now()
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=50)
        end_time = datetime.now()
        print(f"Execution time: {end_time - start_time}")
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(generated_text)

        return generated_text
    
    def episode_generate_widget_captioning(self, image_list, taskDesc_list, current_activity_list, device_id=0) -> List[str]:
        # Prepare the arguments for parallel processing
        results = []
        for i in range(len(image_list)):
            results.append(self.generate_widget_captioning(image_list[i], taskDesc_list[i], current_activity_list[i], device_id))
        return results
    
    
class screen2words:
    def __init__(self) -> None:
        # Load the second processor and model on the second set of GPUs
        self.processor_s2w = AutoProcessor.from_pretrained("google/pix2struct-screen2words-large")
        self.model_s2w = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-screen2words-large").to(self.device)
        # (device_ids_model2[0])
        # model_s2w = DataParallel(model_s2w, device_ids=device_ids_model2)  # Wrap the model with DataParallel for the second set of GPUs
        self.model_s2w = DataParallel(self.model_s2w)

    def generate_screen2words(self, image):
        # conditional generation
        text = "Summarize the function of the screen"
        inputs = self.processor_s2w(text=text, images=image, return_tensors="pt", add_special_tokens=False,is_vqa=True).to(self.device)
        # (device_ids_model2[0])
        with torch.no_grad():
            generated_ids = self.model_s2w.module.generate(**inputs, max_new_tokens=50)
        generated_text = self.processor_s2w.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(generated_text)
    