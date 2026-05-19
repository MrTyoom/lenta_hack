import cv2
import numpy as np

class SmartPriceTagClassifierGist:
    def __init__(self, threshold_percent=4.0):
        self.threshold = threshold_percent

    def predict(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return {"final_decision": "error"}
            
        h, w = image.shape[:2]
        
        crop = image[int(h * 0.40):int(h * 0.95), int(w * 0.05):int(w * 0.95)]
        total_pixels = crop.shape[0] * crop.shape[1]
        
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        colored_mask = cv2.inRange(hsv, np.array([0, 35, 40]), np.array([180, 255, 255]))

        hist_hue = cv2.calcHist([hsv], [0], colored_mask, [180], [0, 180])
        
        hist_percent = (hist_hue / total_pixels) * 100

        yellow_percent = float(np.sum(hist_percent[25:50]))
        red_percent = float(np.sum(hist_percent[0:15])) + float(np.sum(hist_percent[160:180]))

        magenta_shadows = float(np.sum(hist_percent[130:165]))

        if magenta_shadows > 2.0:
            red_percent += magenta_shadows

        blue_percent = float(np.sum(hist_percent[90:120]))

        white_percent = round(100.0 - (yellow_percent + red_percent + blue_percent), 2)

        yellow_percent = round(yellow_percent, 2)
        red_percent = round(red_percent, 2)
        blue_percent = round(blue_percent, 2)

        if red_percent > self.threshold and red_percent > yellow_percent and red_percent > blue_percent:
            final_decision = "red"
                
        elif blue_percent > self.threshold and blue_percent > yellow_percent:
            if blue_percent < 8.0 and white_percent > 80.0:
                final_decision = "white"
            elif red_percent > 5.0:
                final_decision = "red"
            elif yellow_percent > 10.0:
                final_decision = "yellow"
            else:
                final_decision = "blue"

                
        elif yellow_percent > self.threshold:
            if yellow_percent < 8.0 and white_percent > 85.0:
                final_decision = "white"
            else:
                final_decision = "yellow"
        else:
            final_decision = "white"

        avg_h = cv2.mean(hsv)[0]
        avg_s = cv2.mean(hsv)[1]
        avg_v = cv2.mean(hsv)[2]

        return {
            "yellow_confidence": yellow_percent,
            "red_confidence": red_percent,
            "blue_confidence": blue_percent,
            "white_confidence": white_percent,
            "final_decision": final_decision,
            "avg_h": avg_h,
            "avg_s": avg_s,
            "avg_v": avg_v
        }
        

def process_dataset(crops, threshold_percent=4.0):
    """
    crops: список кортежей [(track_id, crop_image), ...]
           где crop_image - numpy array в формате BGR
    возвращает: Dict[int, str] - словарь {track_id: 'color'}
    """
    classifier = SmartPriceTagClassifierGist(threshold_percent=threshold_percent)
    results = {}

    for track_id, crop_image in crops:
        res = classifier.predict(crop_image)
        results[track_id] = res['final_decision']

    return results


# if __name__ == "__main__":
#     base_directory = "best_crops"
#     output_json = "price_tags_results.json"
    
#     process_dataset(base_directory, output_json)