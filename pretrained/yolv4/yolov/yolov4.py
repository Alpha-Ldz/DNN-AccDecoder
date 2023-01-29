from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import check_img_size

def init(weight='yolov4-p5.pt' ,device_='', imgsz=640, batch_size=16):

    device = select_device(device_, batch_size)

    # Load model
    model = attempt_load(weight, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

    # Half
    half = device.type != 'cpu'  # half precision only supported on CUDA

    if half:
        model.half()

    model.eval()

    return model