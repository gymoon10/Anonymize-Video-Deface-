#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

import datetime
import os
import imageio
import imageio.plugins.ffmpeg
import numpy as np
import cv2
import skimage.draw
from collections import Counter

default_onnx_path = '/home/nia_data/processed/temp/centerface.onnx'

class CenterFace:
    def __init__(self, onnx_path=None, in_shape=None, backend='onnxrt'):
        self.in_shape = in_shape
        self.onnx_input_name = 'input.1'
        self.onnx_output_names = ['537', '538', '539', '540']

        if onnx_path is None:
            onnx_path = default_onnx_path

        if backend == 'auto':
            try:
                import onnx
                import onnxruntime
                backend = 'onnxrt'
            except:
                # TODO: Warn when using a --verbose flag
                # print('Failed to import onnx or onnxruntime. Falling back to slower OpenCV backend.')
                backend = 'opencv'
        self.backend = backend


        if self.backend == 'opencv':
            self.net = cv2.dnn.readNetFromONNX(onnx_path)
        elif self.backend == 'onnxrt':
            import onnx
            import onnxruntime

            # Silence warnings about unnecessary bn initializers
            onnxruntime.set_default_logger_severity(3)

            static_model = onnx.load(onnx_path)
            dyn_model = self.dynamicize_shapes(static_model)
            self.sess = onnxruntime.InferenceSession(dyn_model.SerializeToString())

            preferred_provider = self.sess.get_providers()[0]
            preferred_device = 'GPU' if preferred_provider.startswith('CUDA') else 'CPU'
            # print(f'Running on {preferred_device}.')

    @staticmethod
    def dynamicize_shapes(static_model):
        from onnx.tools.update_model_dims import update_inputs_outputs_dims

        input_dims, output_dims = {}, {}
        for node in static_model.graph.input:
            dims = [d.dim_value for d in node.type.tensor_type.shape.dim]
            input_dims[node.name] = dims
        for node in static_model.graph.output:
            dims = [d.dim_value for d in node.type.tensor_type.shape.dim]
            output_dims[node.name] = dims
        input_dims.update({
            'input.1': ['B', 3, 'H', 'W']  # RGB input image
        })
        output_dims.update({
            '537': ['B', 1, 'h', 'w'],  # heatmap
            '538': ['B', 2, 'h', 'w'],  # scale
            '539': ['B', 2, 'h', 'w'],  # offset
            '540': ['B', 10, 'h', 'w']  # landmarks
        })
        dyn_model = update_inputs_outputs_dims(static_model, input_dims, output_dims)
        return dyn_model

    def __call__(self, img, threshold=0.5):
        self.orig_shape = img.shape[:2]
        if self.in_shape is None:
            self.in_shape = self.orig_shape[::-1]
        if not hasattr(self, 'h_new'):  # First call, need to compute sizes
            self.w_new, self.h_new, self.scale_w, self.scale_h = self.transform(self.in_shape)

        blob = cv2.dnn.blobFromImage(
            img, scalefactor=1.0, size=(self.w_new, self.h_new),
            mean=(0, 0, 0), swapRB=False, crop=False
        )
        if self.backend == 'opencv':
            self.net.setInput(blob)
            heatmap, scale, offset, lms = self.net.forward(self.onnx_output_names)
        elif self.backend == 'onnxrt':
            heatmap, scale, offset, lms = self.sess.run(self.onnx_output_names, {self.onnx_input_name: blob})
        else:
            raise RuntimeError(f'Unknown backend {self.backend}')
        dets, lms = self.decode(heatmap, scale, offset, lms, (self.h_new, self.w_new), threshold=threshold)
        if len(dets) > 0:
            dets[:, 0:4:2], dets[:, 1:4:2] = dets[:, 0:4:2] / self.scale_w, dets[:, 1:4:2] / self.scale_h
            lms[:, 0:10:2], lms[:, 1:10:2] = lms[:, 0:10:2] / self.scale_w, lms[:, 1:10:2] / self.scale_h
        else:
            dets = np.empty(shape=[0, 5], dtype=np.float32)
            lms = np.empty(shape=[0, 10], dtype=np.float32)

        return dets, lms

    def transform(self, in_shape):
        h_orig, w_orig = self.orig_shape
        w_new, h_new = in_shape
        # Make spatial dims divisible by 32
        w_new, h_new = int(np.ceil(w_new / 32) * 32), int(np.ceil(h_new / 32) * 32)
        scale_w, scale_h = w_new / w_orig, h_new / h_orig
        return w_new, h_new, scale_w, scale_h

    def decode(self, heatmap, scale, offset, landmark, size, threshold=0.1):
        heatmap = np.squeeze(heatmap)
        scale0, scale1 = scale[0, 0, :, :], scale[0, 1, :, :]
        offset0, offset1 = offset[0, 0, :, :], offset[0, 1, :, :]
        c0, c1 = np.where(heatmap > threshold)
        boxes, lms = [], []
        if len(c0) > 0:
            for i in range(len(c0)):
                s0, s1 = np.exp(scale0[c0[i], c1[i]]) * 4, np.exp(scale1[c0[i], c1[i]]) * 4
                o0, o1 = offset0[c0[i], c1[i]], offset1[c0[i], c1[i]]
                s = heatmap[c0[i], c1[i]]
                x1, y1 = max(0, (c1[i] + o1 + 0.5) * 4 - s1 / 2), max(0, (c0[i] + o0 + 0.5) * 4 - s0 / 2)
                x1, y1 = min(x1, size[1]), min(y1, size[0])
                boxes.append([x1, y1, min(x1 + s1, size[1]), min(y1 + s0, size[0]), s])
                lm = []
                for j in range(5):
                    lm.append(landmark[0, j * 2 + 1, c0[i], c1[i]] * s1 + x1)
                    lm.append(landmark[0, j * 2, c0[i], c1[i]] * s0 + y1)
                lms.append(lm)
            boxes = np.asarray(boxes, dtype=np.float32)
            lms = np.asarray(lms, dtype=np.float32)
            keep = self.nms(boxes[:, :4], boxes[:, 4], 0.3)
            boxes = boxes[keep, :]
            lms = lms[keep, :]
        return boxes, lms

    @staticmethod
    def nms(boxes, scores, nms_thresh):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = np.argsort(scores)[::-1]
        num_detections = boxes.shape[0]
        suppressed = np.zeros((num_detections,), dtype=np.uint8)
        for _i in range(num_detections):
            i = order[_i]
            if suppressed[i]:
                continue
            ix1 = x1[i]
            iy1 = y1[i]
            ix2 = x2[i]
            iy2 = y2[i]
            iarea = areas[i]

            for _j in range(_i + 1, num_detections):
                j = order[_j]
                if suppressed[j]:
                    continue

                xx1 = max(ix1, x1[j])
                yy1 = max(iy1, y1[j])
                xx2 = min(ix2, x2[j])
                yy2 = min(iy2, y2[j])
                w = max(0, xx2 - xx1)
                h = max(0, yy2 - yy1)

                inter = w * h
                ovr = inter / (iarea + areas[j] - inter)
                if ovr >= nms_thresh:
                    suppressed[j] = True
        keep = np.nonzero(suppressed == 0)[0]
        return keep


# In[2]:


def scale_bb(x1, y1, x2, y2, mask_scale):
    s = mask_scale - 1.0
    h, w = y2 - y1, x2 - x1
    y1 -= 0
    y2 += h * s
    x1 -= w * s
    x2 += w * s
    return np.round([x1, y1, x2, y2]).astype(int)


# In[ ]:


child = 'D024'
ipath = f'/home/nia_data/processed/core/assembly/{child}'
anonymize_path = f'/home/nia_data/processed/temp/{child}'
file_list = os.listdir(ipath)

for file in file_list:
    root, ext = os.path.splitext(file)
    video = root.split('/')[-1]
    opath = f'{root}_anonymized{ext}'
    opath2 = f'{root}_anonymized2{ext}'

    if (os.path.isfile(f'{anonymize_path}/{root}_anonymized{ext}')):
        continue

    
    reader: imageio.plugins.ffmpeg.FfmpegFormat.Reader = imageio.get_reader(f'{ipath}/{file}')
    meta = reader.get_meta_data()

    read_iter = reader.iter_data()
    nframes = reader.count_frames()

    writer: imageio.plugins.ffmpeg.FfmpegFormat.Writer = imageio.get_writer(
                  f'{anonymize_path}/{root}_anonymized{ext}', format='FFMPEG', mode='I', fps=meta['fps'], **{"codec": "libx264"} )
        
    print(file, '비식별화 진행')
    centerface = CenterFace(in_shape=None, backend='auto')

    total_faces = []
    for frame in read_iter:  
        dets, _ = centerface(frame, threshold=0.2)
        face = []
        for i, det in enumerate(dets):
            boxes, score = det[:4], det[4]
            x1, y1, x2, y2 = boxes.astype(int)
            x1, y1, x2, y2 = scale_bb(x1, y1, x2, y2, mask_scale=1.03)
            y1, y2 = max(0, y1), min(frame.shape[0] - 1, y2)
            x1, x2 = max(0, x1), min(frame.shape[1] - 1, x2)
            face.append([x1, y1, x2, y2])
            
        total_faces.append(face)

        for idx, f in enumerate(face):
            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]
        
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 0), -1)
        
        writer.append_data(frame)
    
    reader.close()
    writer.close()

    print(file, '비식별화 완료')
    
    faces_detected = ([len(f) for f in total_faces])  # 매 frame당 검출된 얼굴의 수 list
    cnt = Counter(faces_detected) 
    mode = cnt.most_common(1)[0][0]  # 최빈값

    good = {i : f for i, f in enumerate(total_faces) if len(f)==mode}  # 제대로 검출한 good case
    missed = {i : f for i, f in enumerate(total_faces) if len(f)<mode}  # 제대로 검출하지 못한 missed case

    full_detected = list(good.keys())


    print(file, 'interpolation 진행')
    reader: imageio.plugins.ffmpeg.FfmpegFormat.Reader = imageio.get_reader( f'{anonymize_path}/{root}_anonymized{ext}')
    meta = reader.get_meta_data()

    read_iter = reader.iter_data()
    nframes = reader.count_frames()

    writer: imageio.plugins.ffmpeg.FfmpegFormat.Writer = imageio.get_writer(
               f'{anonymize_path}/{root}_anonymized2{ext}', format='FFMPEG', mode='I', fps=meta['fps'], **{"codec": "libx264"} )

    revised = {}
    for key, faces in missed.items():
        low = key 
        up = key 
    
        if low <= min(good.keys()):
            low =  min(good.keys())  # 얼굴을 정상적으로 검출하지 못한 frmae의 index가 정상 케이스 frmae의 index보다 낮으면 (무한 루프 방지)
    
        not_found = True
        while not_found:
            if low in full_detected:
                low = low
                not_found = False
            else:
                low -= 1
                not_found=True
            
        if up >= max(good.keys()):  # 얼굴을 정상적으로 검출하지 못한 frmae의 index가 정상 케이스 frmae의 index보다 높으면
            up = max(good.keys())
    
        else:
            not_found = True
            while not_found:
                if up in full_detected:
                    up = up
                    not_found = False
                else:
                    up += 1
                    not_found=True
            
        low_faces = good[low] 
        up_faces = good[up] 
    
        detection = []
        adjusted = np.round((np.array(low_faces) + np.array(up_faces)) / 2)  
        adjusted = adjusted.astype(int)  # 가장 인접해있는 정상 전,후 frame 2개에서 검출된 얼굴 좌표의 평균 
    
        for i in range(adjusted.shape[0]):
            points = list(adjusted[i]) 
            detection.append(points)
        
        revised[key] = detection
     
    idx = -1
    missed_frame = list(missed.keys())
    for frame in read_iter:
        idx += 1
        if idx in missed_frame:
            additional = missed[idx]
            for f in additional:
                (startX, startY) = int(f[0]), int(f[1])
                (endX, endY) = int(f[2]), int(f[3])
        
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 0), -1)
            
            writer.append_data(frame)
            
            
        else:
            writer.append_data(frame)
    
    reader.close()
    writer.close()
    
    print(file, 'interpolation 종료')

    
   



