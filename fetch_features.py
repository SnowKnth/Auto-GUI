import action_type, action_matching
import numpy as np
from tqdm import tqdm
import json
import jax.numpy as jnp
import argparse
import pickle
import torch
import tensorflow as tf
from PIL import Image
from transformers import AutoProcessor, Blip2Model
import visualization_utils
import pix2struct_caption
import difflib
import logging
import os

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
# model.to(device)
# processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
widget_captioning = pix2struct_caption.widget_captioning()

# 配置日志记录器
logging.basicConfig(
    filename='fetch_features.log',  # 日志文件名
    level=logging.INFO,           # 日志级别
    format='%(asctime)s %(levelname)s %(message)s',  # 日志格式
)

# Set the GPU allocator environment variable
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

dataset_directories = {
    'general': 'android-in-the-wild/general/*',
    'google_apps': 'gs://gresearch/android-in-the-wild/google_apps/*',
    'install': 'android-in-the-wild/install/*',
    'single': 'android-in-the-wild/single/*',
    'web_shopping': 'android-in-the-wild/web_shopping/*',
}

# dataset_directories = {
#     'general': 'dataset/android-in-the-wild/general/*',
#     'google_apps': 'dataset/android-in-the-wild/google_apps/*',
# }

def _decode_image(
    example,
    image_height,
    image_width,
    image_channels,
):
    """Decodes image from example and reshapes.

    Args:
        example: Example which contains encoded image.
        image_height: The height of the raw image.
        image_width: The width of the raw image.
        image_channels: The number of channels in the raw image.

    Returns:
        Decoded and reshaped image tensor.
    """
    image = tf.io.decode_raw(
        example.features.feature['image/encoded'].bytes_list.value[0],
        out_type=tf.uint8,
    )

    height = tf.cast(image_height, tf.int32)
    width = tf.cast(image_width, tf.int32)
    n_channels = tf.cast(image_channels, tf.int32)
    
    # 将图像转换到CPU上进行后续处理
    with tf.device('/CPU:0'):
        image = tf.identity(image)

    return tf.reshape(image, (height, width, n_channels))

def parse_episode(
    episode,
    get_images=False,
    get_annotations=False,
    get_actions=False,
):
    """
    Parses an episode and extracts relevant information based on the specified options.

    Args:
        episode (list): List of examples representing an episode.
        get_images (bool, optional): Whether to extract image features. Defaults to False.
        get_annotations (bool, optional): Whether to extract UI annotations. Defaults to False.
        get_actions (bool, optional): Whether to extract action information. Defaults to False.

    Returns:
        list: List of dictionaries representing the parsed episode. Each dictionary contains the extracted information.

    """
    parsed_episode = []
    for i, ex in enumerate(episode):
        goal = ex.features.feature['goal_info'].bytes_list.value[0].decode('utf-8') #the natural language instruction the episode is demonstrating
        step_id = ex.features.feature['step_id'].int64_list.value[0] # the example's zero-indexed step number within the episode (i.e. if step_id is 2, then this is the third step of the episode)
        
        output_ep = {
            "goal": goal,
            "step_id": step_id
        }

        image_height = ex.features.feature['image/height'].int64_list.value[0]
        image_width = ex.features.feature['image/width'].int64_list.value[0]
        image_channels = ex.features.feature['image/channels'].int64_list.value[0]
        
        if get_images:
            image = _decode_image(ex, image_height, image_width, image_channels)
            image = image.numpy()
            image = Image.fromarray(image).convert('RGB')

            with torch.no_grad():
                inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
                image_features = model.get_image_features(**inputs).pooler_output[0]
                image_features = image_features.detach().cpu()
            output_ep["image"] = image_features

        if get_annotations:
            flattened_positions = np.array(
            ex.features.feature['image/ui_annotations_positions'].float_list.value
            )#a flattened array of coordinates representing the bounding boxes of the UI annotations; the coordinates are in (y, x, height, width) format and the length of this array is 4 * num_elements
            ui_text = ex.features.feature['image/ui_annotations_text'].bytes_list.value #the OCR-detected text associated with the UI element
            ui_text = [value.decode('utf-8') for value in ui_text]
            ui_type = ex.features.feature['image/ui_annotations_ui_types'].bytes_list.value #the type of UI element for each annotation, can be an icon or just text
            ui_type = [value.decode('utf-8') for value in ui_type]

            positions = np.reshape(flattened_positions, (-1, 4)) #(y, x, height, width)
            output_ep["ui_positions"] = positions
            output_ep["ui_text"] = ui_text
            output_ep["ui_type"] = ui_type
        
        if get_actions:
            touch_y, touch_x = ex.features.feature['results/yx_touch'].float_list.value
            lift_y, lift_x = ex.features.feature['results/yx_lift'].float_list.value
            ex_action_type = ex.features.feature['results/action_type'].int64_list.value[0]

            ex_action_type = action_type.ActionType(ex_action_type).name

            type_text = (ex.features.feature['results/type_action'].bytes_list.value[0].decode('utf-8'))
            
            output_ep["result_touch_yx"] = [touch_y, touch_x]
            output_ep["result_lift_yx"] = [lift_y, lift_x]
            output_ep["result_action"] = [ex_action_type, type_text]

        parsed_episode.append(output_ep)
    return parsed_episode



def fetch_episode(dataset_name, data_split, get_images, get_annotations, get_actions):
    filenames = tf.io.gfile.glob(dataset_directories[dataset_name])
    dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP').as_numpy_iterator()

    with open (data_split, "r") as rp:
        split_data = json.load(rp)
        train_data = split_data["train"]
        val_data = split_data["val"]
        test_data = split_data["test"]
        print(f"train_data size: {len(train_data)}, val_data size: {len(val_data)}, test_data size: {len(test_data)}")

    all_parsed_episode = {
        "train": [],
        "val": [],
        "test": [],
    }
    total_screens = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    episode = []
    episode_id = None
    
    for d in tqdm(dataset):
        ex = tf.train.Example()
        ex.ParseFromString(d)
        ep_id = ex.features.feature['episode_id'].bytes_list.value[0].decode('utf-8')
        # if (ep_id not in train_data) & (ep_id not in test_data):
        #     continue
        if episode_id is None:
            episode_id = ep_id
            episode.append(ex)
        elif ep_id == episode_id:
            episode.append(ex)
        else:
            # save data
            try:
                output = parse_episode(episode, get_images=get_images, get_annotations=get_annotations, get_actions=get_actions)
            except Exception as exc:
                print(exc)
                #  bad data point; init a new episode
                episode_id = ep_id
                episode = [ex]

            if episode_id in train_data:
                curr_split = "train"
            elif episode_id in val_data:
                curr_split = "val"
            elif episode_id in test_data:
                curr_split = "test"
            else:
                assert "error episode"
            
            all_parsed_episode[curr_split].append({"episode_id":episode_id, "data":output})
            total_screens[curr_split] += len(episode)
            # init a new episode
            episode_id = ep_id
            episode = [ex]
    # last episode
    if len(episode) > 0:
        # save data
        output = parse_episode(episode, get_images=get_images, get_annotations=get_annotations, get_actions=get_actions)
        if episode_id in train_data:
            curr_split = "train"
        elif episode_id in val_data:
            curr_split = "val"
        elif episode_id in test_data:
            curr_split = "test"
        else:
            assert "error episode"
        
        all_parsed_episode[curr_split].append({"episode_id":episode_id, "data":output})
        total_screens[curr_split] += len(episode)

    print(len(all_parsed_episode["train"]), total_screens["train"], len(all_parsed_episode["val"]), total_screens["val"], len(all_parsed_episode["test"]), total_screens["test"])
    return all_parsed_episode



def parse_episode_ori_image(
    episode,
    get_images=False,
    get_annotations=False,
    get_actions=False,
):
    """
    Parses an episode and extracts relevant information based on the specified options.

    Args:
        episode (list): List of examples representing an episode.
        get_images (bool, optional): Whether to extract image features. Defaults to False.
        get_annotations (bool, optional): Whether to extract UI annotations. Defaults to False.
        get_actions (bool, optional): Whether to extract action information. Defaults to False.

    Returns:
        list: List of dictionaries representing the parsed episode. Each dictionary contains the extracted information.

    """
    parsed_episode = []
    for i, ex in enumerate(episode):
        goal = ex.features.feature['goal_info'].bytes_list.value[0].decode('utf-8') #the natural language instruction the episode is demonstrating
        step_id = ex.features.feature['step_id'].int64_list.value[0] # the example's zero-indexed step number within the episode (i.e. if step_id is 2, then this is the third step of the episode)
        current_activity = ex.features.feature['current_activity'].bytes_list.value[0].decode('utf-8')
        ep_id = ex.features.feature['episode_id'].bytes_list.value[0].decode('utf-8')
        
        output_ep = {
            "goal": goal,
            "step_id": step_id,
            "current_activity": current_activity,
            "episode_id": ep_id
        }

        image_height = ex.features.feature['image/height'].int64_list.value[0]
        image_width = ex.features.feature['image/width'].int64_list.value[0]
        image_channels = ex.features.feature['image/channels'].int64_list.value[0]
        
        output_ep['height_width_channels'] = [image_height, image_width, image_channels]
        
        if get_images:
            image = _decode_image(ex, image_height, image_width, image_channels)
            output_ep["image"] = image

        if get_annotations:
            flattened_positions = np.array(
            ex.features.feature['image/ui_annotations_positions'].float_list.value
            )#a flattened array of coordinates representing the bounding boxes of the UI annotations; the coordinates are in (y, x, height, width) format and the length of this array is 4 * num_elements
            ui_text = ex.features.feature['image/ui_annotations_text'].bytes_list.value #the OCR-detected text associated with the UI element
            ui_text = [value.decode('utf-8') for value in ui_text]
            ui_type = ex.features.feature['image/ui_annotations_ui_types'].bytes_list.value #the type of UI element for each annotation, can be an icon or just text
            ui_type = [value.decode('utf-8') for value in ui_type]

            positions = np.reshape(flattened_positions, (-1, 4)) #(y, x, height, width)
            output_ep["ui_positions"] = positions
            output_ep["ui_text"] = ui_text
            output_ep["ui_type"] = ui_type
        
        if get_actions:
            touch_y, touch_x = ex.features.feature['results/yx_touch'].float_list.value
            lift_y, lift_x = ex.features.feature['results/yx_lift'].float_list.value
            ex_action_type = ex.features.feature['results/action_type'].int64_list.value[0]

            ex_action_type = action_type.ActionType(ex_action_type).name

            type_text = (ex.features.feature['results/type_action'].bytes_list.value[0].decode('utf-8'))
            
            output_ep["result_touch_yx"] = [touch_y, touch_x]
            output_ep["result_lift_yx"] = [lift_y, lift_x]
            output_ep["result_action"] = [ex_action_type, type_text]

        parsed_episode.append(output_ep)
    return parsed_episode


def fetch_episode_ori_image(dataset_name, get_images, get_annotations, get_actions):
        # 假设您知道您的系统有多少个 CPU 核心，您可以设置一个并行读取数目
    num_parallel_reads = 1  # 或者根据您的系统情况进行调整
    filenames = tf.io.gfile.glob(dataset_directories[dataset_name])
        # Filter out temporary or incomplete files
    # filenames = [f for f in filenames if not f.endswith('.gstmp')]
    dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP', num_parallel_reads=num_parallel_reads).as_numpy_iterator()

    episode = []
    episode_id = None
    total_screens = 0
    all_parsed_episode = {}
    
    try:
        for index, d in enumerate(dataset):
          try:
            ex = tf.train.Example()
            ex.ParseFromString(d)
            ep_id = ex.features.feature['episode_id'].bytes_list.value[0].decode('utf-8')
            print(ep_id)
            # if (ep_id not in train_data) & (ep_id not in test_data):
            #     continue
            if episode_id is None:
                episode_id = ep_id
                episode.append(ex)
            elif ep_id == episode_id:
                episode.append(ex)
            else:
                # save data
                try:
                    output = parse_episode_ori_image(episode, get_images=get_images, get_annotations=get_annotations, get_actions=get_actions)
                except Exception as exc:
                    print(exc)
                    #  bad data point; init a new episode
                print(f"Episode {index}\n")
                all_parsed_episode[episode_id] = output
                total_screens += len(episode)
                # init a new episode
                episode_id = ep_id
                episode = [ex]
                
                # 清理内存
                del output
                tf.keras.backend.clear_session()
                tf.compat.v1.reset_default_graph()

          except tf.errors.DataLossError:
            print(f"DataLossError at index {index}, skipping record.")
            logging.error(f"DataLossError at index {index}, skipping record.")
            continue
    except KeyboardInterrupt:
        print("fetch_episode_ori_image检测到 Ctrl+C 中断！")
        return all_parsed_episode
    except Exception as exc:
        logging.error(f"Error: {exc}")
        print(f"Error: {exc}")
        return all_parsed_episode
    
    return all_parsed_episode





def get_episode_and_print_first(dataset_name):
  """Grabs the first complete episode."""
  filenames = tf.io.gfile.glob(dataset_directories[dataset_name])
  raw_dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP').as_numpy_iterator()
  episode = []
  episode_id = None
  for d in raw_dataset:
    ex = tf.train.Example()
    ex.ParseFromString(d)
    ep_id = ex.features.feature['episode_id'].bytes_list.value[0].decode('utf-8')
    if episode_id is None:
      episode_id = ep_id
      episode.append(ex)
    elif ep_id == episode_id:
      episode.append(ex)
    else:
      break
  visualization_utils.plot_episode(episode, show_annotations=True, show_actions=True)
  return episode

def get_episode_and_print_using_episode_id(dataset_name, find_id):
  """Grabs complete episode."""
  filenames = tf.io.gfile.glob(dataset_directories[dataset_name])
  raw_dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP').as_numpy_iterator()
  episode = []
  episode_id = None
  count = 0
  index = 0
  try:  
    for d in raw_dataset:
        index += 1
        ex = tf.train.Example()
        ex.ParseFromString(d)
        ep_id = ex.features.feature['episode_id'].bytes_list.value[0].decode('utf-8')
        # print(index)
        # print(ep_id)
        if ep_id != find_id and episode_id is None:
            continue
        if episode_id is None:
            episode_id = ep_id
            episode.append(ex)
        elif ep_id == episode_id:
            episode.append(ex)
        else:
            break
        
  except tf.errors.DataLossError as e:
        print(f"DataLossError encountered: {e}")
        # 记录错误或其他处理逻辑
        # continue
  except Exception as exc:
        print(exc)
        # continue
  if episode.__len__() == 0:
      print(f"Episode {find_id} not found")
  visualization_utils.plot_episode(episode, show_annotations=True, show_actions=True)
  return episode

def get_episode_and_generate_widget_caption(dataset_name, episode_search):
    '''Edit... to search for a specific episode_id in original dataset and generate widget captioning'''
    with open(f"../dataset_Auto-UI/ori_image/{dataset_name}_withImage.obj", "rb") as read_file:
        all_parsed_episode = pickle.load(read_file)
    wc = pix2struct_caption.widget_captioning()
    episode_id = None
    episode = []
    # print("***test: ", all_parsed_episode)
    for key, value in all_parsed_episode.items():  # Add 'enumerate' to loop over the episodes
        episode_id = key
        episode = value
        # print(f'{episode_id:}\n')
        if episode_id == episode_search:
            print(f'{episode_id:}\n')
            for index,ep in enumerate(episode):
                generate_widget_caption(ep, wc, index)
                
def generate_widget_caption(episode_trans, widget_captioning, index):
    '''Generate widget caption for episode''' 
    np_image = episode_trans['image'].numpy()
    pil_image = Image.fromarray(np_image)
    image = pil_image.convert("RGB")
    width, height = image.size
    goal = episode_trans['goal']
    ep_id = episode_trans['episode_id']
    current_act = episode_trans['current_activity']
    # ep['activity']
    box = get_smaller_widget_box(episode_trans['result_touch_yx'], episode_trans['ui_positions'], width, height)
    image = widget_captioning.mark_widget_for_wc(bbox=box,image=image,saveas_id=f'{ep_id}_{index}')
    gen_text = widget_captioning.generate_widget_captioning(image, taskDesc=goal, current_activity=current_act)
    return gen_text

               
def generate_widget_caption_for_episode_search_id(dataset_name, episode_search_id):
    '''Function the same as get_episode_and_generate_widget_caption but more efficient'''
    with open(f"dataset/episode/{dataset_name}_withImage.obj", "rb") as read_file:
        all_parsed_episode = pickle.load(read_file)
    wc = pix2struct_caption.widget_captioning()
    episode = all_parsed_episode[episode_search_id]
    visualization_utils.plot_episode_from_stored(episode_search_id, episode, show_annotations=True, show_actions=True)
    for index,ep in enumerate(episode):
        np_image = ep['image'].numpy()
        pil_image = Image.fromarray(np_image)
        image = pil_image.convert("RGB")
        width, height = image.size
        goal = ep['goal']
        current_act = f'activity for {goal}'
        # ep['activity']
        box = get_smaller_widget_box(ep['result_touch_yx'], ep['ui_positions'], width, height)
        image = wc.mark_widget_for_wc(bbox=box,image=image,saveas_id=index)
        wc.generate_widget_captioning(image, taskDesc=goal, current_activity=current_act)


                
def generate_widget_caption_from_stored(dataset_name):
    with open(f"dataset/ori_image/{dataset_name}_withImage.obj", "rb") as read_file:
        all_parsed_episode = pickle.load(read_file)
    wc = pix2struct_caption.widget_captioning()
    episode_id = None
    episode = []
    for key, value in all_parsed_episode.items():  # Add 'enumerate' to loop over the episodes
        episode_id = key
        episode = value
        for index,ep in enumerate(episode):
            np_image = ep['image'].numpy()
            pil_image = Image.fromarray(np_image)
            image = pil_image.convert("RGB")
            width, height = image.size
            goal = ep['goal']
            # current_activity = ep['activity']
            box = get_smaller_widget_box(ep['result_touch_yx'], ep['ui_positions'], width, height)
            image = wc.mark_widget_for_wc(bbox=box,image=image,saveas_id=index)
            wc.generate_widget_captioning(image, taskDesc=goal)
            get_episode_and_print_using_episode_id('general', episode_id)

                
def get_episode_and_describe_from_stored(dataset_name):
    with open(f"dataset/episode/{dataset_name}_withImage.obj", "rb") as read_file:
        all_parsed_episode = pickle.load(read_file)
    episode_id = None
    episode = []
    for key, value in all_parsed_episode.items():  # Add 'enumerate' to loop over the episodes
        episode_id = key
        episode = value
            # visualization_utils.plot_episode(episode, show_annotations=True, show_actions=True)
        generate_episode_description(episode_id, episode, dataset_name)
        # visualization_utils.plot_episode(episode, show_annotations=True, show_actions=True)

                
def generate_describe_for_dataset(dataset_name, get_images, get_annotations, get_actions):
    num_parallel_reads = 1  # 或者根据您的系统情况进行调整
    filenames = tf.io.gfile.glob(dataset_directories[dataset_name])
        # Filter out temporary or incomplete files
    # filenames = [f for f in filenames if not f.endswith('.gstmp')]
    dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP', num_parallel_reads=num_parallel_reads).as_numpy_iterator()

    episode = []
    episode_id = None
    total_screens = 0
    
    try:
        for index, d in enumerate(dataset):
          try:
            ex = tf.train.Example()
            ex.ParseFromString(d)
            ep_id = ex.features.feature['episode_id'].bytes_list.value[0].decode('utf-8')
            print(ep_id)
            # if (ep_id not in train_data) & (ep_id not in test_data):
            #     continue
            if episode_id is None:
                episode_id = ep_id
                episode.append(ex)
            elif ep_id == episode_id:
                episode.append(ex)
            else:
                # save data
                try:
                    output = parse_episode_ori_image(episode, get_images=get_images, get_annotations=get_annotations, get_actions=get_actions)
                except Exception as exc:
                    print(exc)
                    #  bad data point; init a new episode
                print(f"Episode {index}\n")
                generate_episode_description(episode_id, output, dataset_name)
                total_screens += len(episode)
                # init a new episode
                episode_id = ep_id
                episode = [ex]
                
                # 清理内x存
                del output
                tf.keras.backend.clear_session()
                tf.compat.v1.reset_default_graph()

          except tf.errors.DataLossError:
            print(f"DataLossError at index {index}, skipping record.")
            logging.error(f"DataLossError at index {index}, skipping record.")
            continue
    except KeyboardInterrupt:
        print("fetch_episode_ori_image检测到 Ctrl+C 中断！")
        return total_screens
    # except Exception as exc:
    #     logging.error(f"Error: {exc}")
    #     print(f"Error: {exc}")
    #     return total_screens
    
    return total_screens


            
def generate_episode_description(episode_id, episode, dataset_name):   
    steps = []
    task_possible = True
    goal = None
    for index, ep in enumerate(episode):
        goal = ep['goal']
        act_type, type_text = ep['result_action']  
        if act_type.lower() == action_type.ActionType.TYPE.name.lower():
            step = f"And type text '{type_text}' into the text field."   
        elif act_type.lower() == action_type.ActionType.DUAL_POINT.name.lower(): 
            if action_matching.is_tap_action(ep['result_touch_yx'], ep['result_lift_yx']):
                step = get_action_widget(episode_id, ep, ep['ui_positions'], ep['result_touch_yx'], index)
            else:
                action_1_touch_yx = jnp.asarray(ep['result_touch_yx'])
                action_1_lift_yx = jnp.asarray(ep['result_lift_yx'])
                drag_1_deltas = action_1_lift_yx - action_1_touch_yx
                drag_1_magnitudes = jnp.abs(drag_1_deltas)
                drag_1_main_axis = np.argmax(drag_1_magnitudes)
                if drag_1_deltas[drag_1_main_axis] < 0 and drag_1_main_axis == 0:
                    direc = "up" 
                if drag_1_deltas[drag_1_main_axis] > 0 and drag_1_main_axis == 0:
                    direc = "down"
                if drag_1_deltas[drag_1_main_axis] < 0 and drag_1_main_axis == 1:
                    direc = "left" 
                if drag_1_deltas[drag_1_main_axis] > 0 and drag_1_main_axis == 1:
                    direc = "right"
                step = f"Drag {direc} to extend the screen content or change widget state."
        elif act_type.lower() == action_type.ActionType.PRESS_BACK.name.lower():
            step = "Press Back"
        elif act_type.lower() == action_type.ActionType.PRESS_HOME.name.lower():
            step = "Press Home"
        elif act_type.lower() == action_type.ActionType.PRESS_ENTER.name.lower():
            step = "Press Enter"
        elif act_type.lower() == action_type.ActionType.STATUS_TASK_COMPLETE.name.lower():
            step = "Task Complete"
        elif act_type.lower() == action_type.ActionType.STATUS_TASK_IMPOSSIBLE.name.lower():
            step = "Task Impossible"
            task_possible = False
        else:
            step = "Unknown Action"
        steps.append(step)
    
    if task_possible:
        data_to_write = {
            "episode_id": episode_id,
            "goal": goal,
            "steps": steps
        }
        filename = f"dataset/steps_desc/{dataset_name}_episode_description.json"
        # 将字典写入JSON文件
        # 确保目录存在
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'a') as file:
            json.dump(data_to_write, file, indent=4)  # indent参数用于美化输出
        print(f"Data written to {filename}")
    else:
        logging.info(f"Task Impossible: {episode_id}")
 
 
def get_smaller_widget_box(touch_yx, positions, width, height):
    '''return bounding box positions covering the nearest widget annotated in AITW'''
    scalar = 0.8 #additional added percent
        # 计算每个中心点与touch_yx的距离，要乘宽高
    def calculate_distance(box, touch_point):
        # 获取中心点坐标
        center_y = (box[0] + (box[0]+box[2])) * height / 2
        center_x = (box[1] + (box[1]+box[3])) * width / 2
        return ((center_y - touch_point[0] * height) ** 2 + (center_x - touch_point[1] * width) ** 2) ** 0.5   
    if positions.shape[0] == 0:
        return [0,0,0,0]
    distances = [
        calculate_distance(positions[i], touch_yx) for i in range(positions.shape[0])
    ]       
    # 找到最小距离的索引
    min_distance_index = distances.index(min(distances))
    target_pos = positions[min_distance_index]
    # Define the bounding box coordinates (left, top, right, bottom)
    tp = [target_pos[1]- target_pos[3]*scalar/2, target_pos[0]-target_pos[2]*scalar/2, target_pos[1] + target_pos[3]*(1+scalar/2),  target_pos[0] + target_pos[2]*(1+scalar/2)]
    y = [np.abs(touch_yx[0]-tp[1]),np.abs(touch_yx[0]-tp[3])]
    x = [np.abs(touch_yx[1]-tp[0]),np.abs(touch_yx[1]-tp[2])]
    max_y = np.max(y)
    max_x = np.max(x)
    
    new_left = touch_yx[1] - max_x
    new_top = touch_yx[0] - max_y
    new_right = touch_yx[1] + max_x
    new_bottom = touch_yx[0] + max_y
    # (left, top, right, bottom)
    return [max(0,new_left)*width, max(0,new_top)*height, min(new_right,1)*width, min(new_bottom,1)*height]
           
    
def get_action_widget(ep_id, ex, positions, touch_yx, index):
    wc_text = generate_widget_caption(ex, widget_captioning, index)
    
    resized_annotation_positions = action_matching.resize_annotation_bounding_boxes(
      positions,
      annotation_width_augment_fraction = action_matching.ANNOTATION_WIDTH_AUGMENT_FRACTION,
      annotation_height_augment_fraction = action_matching.ANNOTATION_HEIGHT_AUGMENT_FRACTION,
    )
    tap1_in_box = action_matching.yx_in_bounding_boxes(touch_yx, resized_annotation_positions)
    # 查找tap1_in_box中值为1的索引
    indices_with_1 = [i for i, x in enumerate(tap1_in_box) if x == 1]

    # 如果没有找到1，则返回None
    if not indices_with_1:
        logging.warning(f"Tap not in any bounding box. episode_id: {ep_id}  step_id: {ex['step_id']}")
        return f"Tap touch_yx: {touch_yx}({wc_text})"
           
    # 如果只有一个1，直接返回对应的resized_annotation_positions
    if len(indices_with_1) == 1:
        return f"Tap {ex['ui_type'][indices_with_1[0]]}('{ex['ui_text'][indices_with_1[0]]},{wc_text}') "
    
    # 计算每个中心点与touch_yx的距离
    def calculate_distance(box, touch_point):
        # 获取中心点坐标
        center_y = (box[0] + (box[0]+box[2])) / 2
        center_x = (box[1] + (box[1]+box[3]))/ 2
        return ((center_y - touch_point[0]) ** 2 + (center_x - touch_point[1]) ** 2) ** 0.5   
    [height,width] = ex['height_width_channels'][:2]
    distances = [
        calculate_distance(resized_annotation_positions[i]*jnp.array([height,width,height,width]), jnp.array(touch_yx)*jnp.array([height,width])) 
        for i in indices_with_1
    ]
    # 找到最小距离的索引
    min_distance_index = indices_with_1[distances.index(min(distances))]   
    # 返回距离最近的bounding box
    logging.warning(f"Multiple expanded bounding box. choose nearest. episode_id: {ep_id}  step_id: {ex['step_id']}")
    return f"Tap ({ex['ui_text'][min_distance_index]},{wc_text}) {ex['ui_type'][min_distance_index]}"

    
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='general')
    parser.add_argument("--split_file", type=str, default="dataset/general_texts_splits.json")
    parser.add_argument('--output_dir', type=str, default='dataset/t5/general_parsed_episode_t5_clip')
    parser.add_argument('--get_images', default=True, action='store_true')
    parser.add_argument('--get_annotations', default=True, action='store_true')
    parser.add_argument('--get_actions', default=True, action='store_true')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    get_episode_and_print_using_episode_id('web_shopping', '4458374961172202455')
    # get_episode_and_print_using_episode_id('general', '7044099285928696243')
    # get_episode_and_print_using_episode_id('general', '15436164102229426615')
    # get_episode_and_print_using_episode_id('general', '4545355796601498876')
    # get_episode_and_print_using_episode_id('single', '17652164182256576003.660-752.Search Brooklyn apartments')
    # get_episode_and_print_using_episode_id('single', '1396884187066474035.819-976.add to cart then go to cart')
    # get_episode_and_print_using_episode_id('single', '17652164182256576003.16-264.Start installing Zillow App')
    #1991801530150611027\7044099285928696243\15436164102229426615\4545355796601498876
    # get_episode_and_describe_from_stored('single') 
    # generate_widget_caption_for_episode_search_id('single', '17652164182256576003.660-752.Search Brooklyn apartments')
    # generate_widget_caption_for_episode_search_id('single', '17652164182256576003.16-264.Start installing Zillow App')
    # generate_widget_caption_from_stored('general')   
    # get_episode_and_generate_widget_caption('general', '1991801530150611027') 
    # get_episode_and_generate_widget_caption('general', '7044099285928696243') 
    # get_episode_and_generate_widget_caption('general', '15436164102229426615') 
    # get_episode_and_generate_widget_caption('general', '14482767708983232687') 
    # get_episode_and_print_first('general')
    # get_episode_and_print_first('google_apps')
    # get_episode_and_print_first('web_shopping')
    # get_episode_and_print_first('install')
    # get_episode_and_print_first('single')
    # generate_widget_caption_for_episode_search_id('general', '1684888717715898644')
    # generate_widget_caption_for_episode_search_id('web_shopping', '7636333411160919632')

    
    
    
    # args = parse_args()
    # print('====Input Arguments====')
    # print(json.dumps(vars(args), indent=2, sort_keys=False))


    # all_parsed_episode = fetch_episode(args.dataset, args.split_file, args.get_images, args.get_annotations, args.get_actions)
    
    # with open(f"{args.output_dir}_train.obj", "wb") as wp:
    #     pickle.dump(all_parsed_episode["train"],wp)
    # with open(f"{args.output_dir}_val.obj", "wb") as wp:
    #     pickle.dump(all_parsed_episode["val"],wp)
    # with open(f"{args.output_dir}_test.obj", "wb") as wp:
    #     pickle.dump(all_parsed_episode["test"],wp)
        
        
        
    # args = parse_args()
    # print('====Input Arguments====')
    # print(json.dumps(vars(args), indent=2, sort_keys=False))
    # all_parsed_episode = fetch_episode_ori_image(args.dataset , args.get_images, args.get_annotations, args.get_actions)
    # with open(f"{args.output_dir}/{args.dataset}_withImage.obj", "wb") as wp:
    #     pickle.dump(all_parsed_episode,wp)
   
    # args = parse_args()
    # print('====Input Arguments====')
    # print(json.dumps(vars(args), indent=2, sort_keys=False))
    # generate_describe_for_dataset(args.dataset, args.get_images, args.get_annotations, args.get_actions)