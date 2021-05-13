import platform

import cv2
import numpy as np
import tensorflow as tf
import tflite_runtime.interpreter as tflite

from postprocessing import nms_oneclass

# EdgeTPU shared lib name
EDGETPU_SHARED_LIB = {
    'Linux': 'libedgetpu.so.1',
    'Darwin': 'libedgetpu.1.dylib',
    'Windows': 'edgetpu.dll'
}[platform.system()]


class BaseInferencer:

    def __init__(self, model_path, edgetpu=True):
        experimental_delegates = [tf.lite.experimental.load_delegate(EDGETPU_SHARED_LIB)] if edgetpu else None
        self.interpreter = tflite.Interpreter(
            model_path=model_path,
            experimental_delegates=
            experimental_delegates)
        self.interpreter.allocate_tensors()
        self.input_idx = self.interpreter.get_input_details()[0]['index']
        self.input_shape = self.interpreter.get_input_details()[0]['shape'][1:3]

    def inference(self, src):
        raise NotImplementedError("inference not implemented!")


class FaceDetector(BaseInferencer):

    SCORE_THRESH = 0.75
    MAX_FACE_NUM = 10

    ANCHOR_STRIDES = [8, 16]
    ANCHOR_NUM = [2, 6]

    def __init__(self, model_path, edgetpu=True):
        super(FaceDetector, self).__init__(model_path, edgetpu)
        self.outputs_idx = {}
        for output in self.interpreter.get_output_details():
            self.outputs_idx[output['name']] = output['index']
        self.anchors = self.create_anchors(self.input_shape)

    def inference(self, image):

        # todo: input type check
        # convert to float32
        image_ = cv2.resize(image, tuple(self.input_shape)).astype(np.float32)
        image_ = (image_ - 128.0) / 128.0
        image_ = image_[None, ...]

        # invoke
        self.interpreter.set_tensor(self.input_idx, image_)
        self.interpreter.invoke()
        scores = self.interpreter.get_tensor(self.outputs_idx['classificators']).squeeze()
        scores = 1 / (1 + np.exp(-scores))
        bboxes = self.interpreter.get_tensor(self.outputs_idx['regressors']).squeeze()

        bboxes_decoded, landmarks, scores = self.decode(scores, bboxes)
        bboxes_decoded *= image.shape[0]
        landmarks *= image.shape[0]

        if len(bboxes_decoded) == 0:
            return np.array([]), np.array([]), np.array([])

        keep_mask = nms_oneclass(bboxes_decoded, scores)  # np.ones(pred_bbox.shape[0]).astype(bool)
        bboxes_decoded = bboxes_decoded[keep_mask]
        landmarks = landmarks[keep_mask]
        scores = scores[keep_mask]

        return bboxes_decoded, landmarks, scores

    def decode(self, scores, bboxes):

        w, h = self.input_shape

        cls_mask = scores > self.SCORE_THRESH
        if cls_mask.sum() == 0:
            return np.array([]), np.array([]), np.array([])

        scores = scores[cls_mask]
        bboxes = bboxes[cls_mask]
        bboxes_anchors = self.anchors[cls_mask]

        bboxes_decoded = bboxes_anchors.copy()
        bboxes_decoded[:, 0] += bboxes[:, 1]  # row
        bboxes_decoded[:, 1] += bboxes[:, 0]  # columns
        bboxes_decoded[:, 0] /= h
        bboxes_decoded[:, 1] /= w

        pred_w = bboxes[:, 2] / w
        pred_h = bboxes[:, 3] / h

        topleft_x = bboxes_decoded[:, 1] - pred_w * 0.5
        topleft_y = bboxes_decoded[:, 0] - pred_h * 0.5
        btmright_x = bboxes_decoded[:, 1] + pred_w * 0.5
        btmright_y = bboxes_decoded[:, 0] + pred_h * 0.5

        pred_bbox = np.stack([topleft_x, topleft_y, btmright_x, btmright_y], axis=-1)

        # decode landmarks
        landmarks = bboxes[:, 4:]
        landmarks[:, 1::2] += bboxes_anchors[:, 0:1]
        landmarks[:, ::2] += bboxes_anchors[:, 1:2]
        landmarks[:, 1::2] /= h
        landmarks[:, ::2] /= w

        return pred_bbox, landmarks, scores

    @classmethod
    def create_anchors(cls, input_shape):
        w, h = input_shape
        anchors = []
        for s, a_num in zip(cls.ANCHOR_STRIDES, cls.ANCHOR_NUM):
            gridCols = (w + s - 1) // s
            gridRows = (h + s - 1) // s
            x, y = np.meshgrid(np.arange(gridRows), np.arange(gridCols))
            x, y = x[..., None], y[..., None]
            anchor_grid = np.concatenate([y, x], axis=-1)
            anchor_grid = np.tile(anchor_grid, (1, 1, a_num))
            anchor_grid = s * (anchor_grid.reshape(-1, 2) + 0.5)
            anchors.append(anchor_grid)
        return np.concatenate(anchors, axis=0)


class FaceMesher(BaseInferencer):

    FACE_KEY_NUM = 468

    def __init__(self, model_path, edgetpu=True):
        super(FaceMesher, self).__init__(model_path, edgetpu)
        outputs_idx_tmp = {}
        for output in self.interpreter.get_output_details():
            outputs_idx_tmp[output['name']] = output['index']
        self.outputs_idx = {'landmark': outputs_idx_tmp['conv2d_20'],
                            'score': outputs_idx_tmp['conv2d_30']}

    def inference(self, image):

        h, w = self.input_shape

        image_ = cv2.resize(image, tuple(self.input_shape)).astype(np.float32)
        image_ = (image_ - 128.0) / 128.0
        if len(image_.shape) < 4:
            image_ = image_[None, ...]

        # invoke
        self.interpreter.set_tensor(self.input_idx, image_)
        self.interpreter.invoke()
        landmarks = self.interpreter.get_tensor(self.outputs_idx['landmark'])
        scores = self.interpreter.get_tensor(self.outputs_idx['score'])

        # postprocessing
        landmarks = landmarks.reshape(self.FACE_KEY_NUM, 3)
        landmarks[:, 0] /= w
        landmarks[:, 1] /= h

        landmarks[:, 0] *= image.shape[1]
        landmarks[:, 1] *= image.shape[0]

        return landmarks, scores


class FaceAligner:
    '''reference to https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/'''

    def __init__(self,
                 desiredLeftEye=(0.35, 0.35),
                 desiredFaceWidth=192,
                 desiredFaceHeight=None):
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight

        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def align(self, image, landmarks):
        landmarks = landmarks.astype(int).reshape(-1, 2)

        # get left and right eye
        left_eye = landmarks[1]
        right_eye = landmarks[0]

        # computer angle
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # compute the location of right/left eye in new image
        desiredRightEyeX = 1 - self.desiredLeftEye[0]

        # get the scale based on the distance
        dist = np.sqrt(dY ** 2 + dX ** 2)
        desired_dist = (desiredRightEyeX - self.desiredLeftEye[0])
        desired_dist *= self.desiredFaceWidth
        scale = desired_dist / (dist + 1e-6)

        # get the center of eyes
        eye_center = (left_eye + right_eye) // 2

        # get transformation matrix
        M = cv2.getRotationMatrix2D(tuple(eye_center), angle, scale)

        # align the center
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eye_center[0])
        M[1, 2] += (tY - eye_center[1])  # update translation vector

        # apply affine transformation
        dst_size = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, dst_size, flags=cv2.INTER_CUBIC)

        return output, M

    @staticmethod
    def inverse(mesh_landmark, M):
        M_inverse = cv2.invertAffineTransform(M)
        px = (M_inverse[0, 0] * mesh_landmark[:, 0:1] + M_inverse[0, 1] * mesh_landmark[:, 1:2] + M_inverse[0, 2])
        py = (M_inverse[1, 0] * mesh_landmark[:, 0:1] + M_inverse[1, 1] * mesh_landmark[:, 1:2] + M_inverse[1, 2])
        mesh_landmark_inverse = np.concatenate([px, py, mesh_landmark[:, 2:]], axis=-1)
        return mesh_landmark_inverse
