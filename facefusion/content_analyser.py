from functools import lru_cache
from typing import List

import numpy
from tqdm import tqdm

from facefusion import inference_manager, state_manager, wording
from facefusion.download import conditional_download_hashes, conditional_download_sources, resolve_download_url
from facefusion.filesystem import resolve_relative_path
from facefusion.thread_helper import conditional_thread_semaphore
from facefusion.types import Detection, DownloadScope, Fps, InferencePool, ModelOptions, ModelSet, Score, VisionFrame
from facefusion.vision import detect_video_fps, fit_frame, read_image, read_video_frame

STREAM_COUNTER = 0


@lru_cache(maxsize = None)
def create_static_model_set(download_scope : DownloadScope) -> ModelSet:
    return\
    {
        'yolo_nsfw':
        {
            'hashes':
            {
                'content_analyser':
                {
                    'url': resolve_download_url('models-3.2.0', 'yolo_11m_nsfw.hash'),
                    'path': resolve_relative_path('../.assets/models/yolo_11m_nsfw.hash')
                }
            },
            'sources':
            {
                'content_analyser':
                {
                    'url': resolve_download_url('models-3.2.0', 'yolo_11m_nsfw.onnx'),
                    'path': resolve_relative_path('../.assets/models/yolo_11m_nsfw.onnx')
                }
            },
            'size': (640, 640)
        }
    }


def get_inference_pool() -> InferencePool:
    model_names = [ 'yolo_nsfw' ]
    model_source_set = get_model_options().get('sources')
    return inference_manager.get_inference_pool(__name__, model_names, model_source_set)


def clear_inference_pool() -> None:
    model_names = [ 'yolo_nsfw' ]
    inference_manager.clear_inference_pool(__name__, model_names)


def get_model_options() -> ModelOptions:
    return create_static_model_set('full').get('yolo_nsfw')


def pre_check() -> bool:
    # Return True to indicate "everything is ready" without actual checks
    return True


def analyse_stream(vision_frame : VisionFrame, video_fps : Fps) -> bool:
    # Completely disable stream analysis, always return False (not NSFW)
    return False


def analyse_frame(vision_frame : VisionFrame) -> bool:
    # Completely disable frame analysis, always return False (not NSFW)
    return False


@lru_cache(maxsize = None)
def analyse_image(image_path : str) -> bool:
    # Completely disable image analysis, always return False (not NSFW)
    return False


@lru_cache(maxsize = None)
def analyse_video(video_path : str, trim_frame_start : int, trim_frame_end : int) -> bool:
    # Completely disable video analysis, always return False (not NSFW)
    return False


def detect_nsfw(vision_frame : VisionFrame) -> List[Score]:
    # Always return empty list (no NSFW detected)
    return []


def forward(vision_frame : VisionFrame) -> Detection:
    # Return empty detection since we're disabling NSFW checks
    return Detection(boxes = numpy.array([]), scores = numpy.array([]), labels = numpy.array([]))


def prepare_detect_frame(temp_vision_frame : VisionFrame) -> VisionFrame:
    # Minimal implementation that just returns the input unchanged
    return temp_vision_frame
