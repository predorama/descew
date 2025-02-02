import streamlit as st
import cv2
import numpy as np
from PIL import Image
from deskew import Deskew
from streamlit_drawable_canvas import st_canvas


def crop_image(image, rect_coords):
    x_min, y_min = np.min(rect_coords, axis=0)
    x_max, y_max = np.max(rect_coords, axis=0)
    cropped_image = image[y_min:y_max, x_min:x_max]
    return cropped_image


def deskew_image(image_array):
    try:
        if image_array is None or image_array.size == 0:
            raise ValueError("Invalid image array provided for deskewing.")
        deskew_obj = Deskew(
            image_array=image_array, display_image=False, output_file=None, r_angle=0
        )
        deskewed_image = deskew_obj.deskew()
        return deskewed_image
    except Exception as e:
        raise RuntimeError(f"Error during deskewing: {e}")


def main():
    st.title("Обрезка, выравнивание и сплющивание изображения")
    st.write("Перетащите изображение, обрежьте его, а затем выровняйте и сплющите.")
    uploaded_file = st.file_uploader(
        "Загрузите изображение", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        try:
            # Чтение файла в виде байтов
            file_bytes = uploaded_file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            if nparr.size == 0:
                st.error(
                    "Загруженный файл пуст. Пожалуйста, загрузите корректное изображение."
                )
                return
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                st.error(
                    "Не удалось декодировать изображение. Пожалуйста, загрузите корректный файл изображения."
                )
                return

            # Масштабирование изображения для отображения в st_canvas с 20% padding
            max_display_size = 780  # Максимальная ширина или высота для отображения
            padding_percent = 0.20  # 20% padding
            height, width = image.shape[:2]

            # Calculate the scaling factor with padding
            scale_factor = min(
                max_display_size / (width * (1 + padding_percent)),
                max_display_size / (height * (1 + padding_percent)),
            )

            # Вычисляем новые размеры для canvas с учетом padding
            canvas_height = int(height * scale_factor)
            canvas_width = int(width * scale_factor)

            # Преобразуем изображение в RGB для отображения в canvas
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb).resize((canvas_width, canvas_height))

            # Отображение уменьшенного изображения для рисования прямоугольника
            st.write("Нарисуйте прямоугольник вокруг области, которую хотите обрезать:")
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=2,
                stroke_color="#FFFFFF",
                background_image=pil_image,
                update_streamlit=True,
                height=canvas_height,
                width=canvas_width,
                drawing_mode="rect",
                key="canvas",
            )

            if (
                canvas_result.json_data is not None
                and len(canvas_result.json_data["objects"]) > 0
            ):
                rect = canvas_result.json_data["objects"][0]
                x_min = int(rect["left"] / scale_factor)
                y_min = int(rect["top"] / scale_factor)
                x_max = int((rect["left"] + rect["width"]) / scale_factor)
                y_max = int((rect["top"] + rect["height"]) / scale_factor)
                rect_coords = [(x_min, y_min), (x_max, y_max)]

                try:
                    cropped_image = crop_image(image, rect_coords)
                    st.image(
                        cropped_image, channels="BGR", caption="Обрезанное изображение"
                    )
                except Exception as e:
                    st.error(f"Ошибка при обрезке изображения: {e}")
                    return

                if st.button("Выровнять и сплющить изображение"):
                    with st.spinner("Обработка изображения..."):
                        try:
                            deskewed_image = deskew_image(cropped_image)
                            if deskewed_image is not None and deskewed_image.size > 0:
                                st.image(
                                    deskewed_image,
                                    channels="BGR",
                                    caption="Выровненное и сплющенное изображение",
                                )

                                # Ensure the deskewed image is in uint8 format
                                if deskewed_image.dtype != np.uint8:
                                    deskewed_image = (deskewed_image * 255).astype(
                                        np.uint8
                                    )  # Convert to uint8

                                # Сохраняем изображение в формате JPEG
                                _, img_encoded = cv2.imencode(".jpg", deskewed_image)
                                st.download_button(
                                    label="Скачать выровненное изображение",
                                    data=img_encoded.tobytes(),
                                    file_name="deskewed_image.jpg",
                                    mime="image/jpeg",
                                )
                            else:
                                st.error(
                                    "Выровненное изображение пусто или некорректно."
                                )
                        except Exception as e:
                            st.error(f"Ошибка при выравнивании изображения: {e}")
            else:
                st.error("Пожалуйста, нарисуйте прямоугольник для обрезки изображения.")
        except Exception as e:
            st.error(f"Произошла ошибка при обработке файла: {e}")


if __name__ == "__main__":
    main()
