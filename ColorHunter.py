from sklearn.cluster import KMeans
import numpy as np
import pyperclip
import GUI
import cv2


class ColorHunter:
    """
    A class to represent the ColorHunter application.
    This application extracts and displays color palettes from images.
    """

    def __init__(self, width, height):
        """
        Initialize the ColorHunter application.

        Parameters:
        width (int): The width of the main window.
        height (int): The height of the main window.
        """
        self.main_window = GUI.Window(width, height, 'Color Hunter', (255, 218, 185),'assets/img/icon.ico')
        self.image_data_list = []
        self.loadUI()

    def loadUI(self):
        """
        Load and set up the User Interface components.
        """
        self.main_layout = GUI.Layout(self.main_window.central, 'main_layout', 'h')
        self.left_layout = GUI.Layout(self.main_layout, 'left_layout', 'v')
        self.right_layout = GUI.Layout(self.main_layout, 'right_layout', 'v', alignment='centertop')
        self.main_layout.share_equal_space()

        self.color_title = GUI.Label(self.right_layout, 'Color pallet', name='pallet_title', alignment='centerbot',
                                     font_size=16)
        self.color_title.setContentsMargins(0, 15, 0, 0)

        self.color_display = GUI.Layout(self.right_layout, 'color_display', 'g', alignment='centertop',
                                        size=GUI.QSize(300, 250))

        self.image_title = GUI.Label(self.right_layout, 'Image preview', name='pallet_title', alignment='centerbot',
                                     font_size=16)

        self.image_display = GUI.Layout(self.right_layout, 'image_display', 'v', alignment='center',
                                        size=GUI.QSize(300, 170), background_color=(255, 255, 255))
        self.preview_image = GUI.Label(self.image_display, '', 'preview_image', alignment='center',
                                       size=GUI.QSize(300, 170))
        self.preview_image.add_style('none', 'background-color', 'rgb(255,255,235)')

        self.list_title = GUI.Label(self.left_layout, 'Image list', name='list_title', alignment='center', font_size=16)
        self.list_title.setFixedHeight(30)

        self.image_list = GUI.List(self.left_layout, name='image_list', size=GUI.QSize(300, 400),
                                   color='rgb(255,255,235)')
        self.image_list.set_item_filter('img')
        self.image_list.set_custom_function('add_item',self.add_image)
        self.image_list.set_custom_function('del_item',self.delete_image)
        self.image_list.set_custom_function('item_changed',self.update_change)


    def add_image(self):
        """
        Add an image to the application, extract its colors, and store the data.
        """
        image = self.get_image()
        image_color_list = self.extract_colors(image)

        self.image_data_list.append([self.cv2qt_image(image), image_color_list])

    def update_change(self):
        """
          Update the displayed image and color palette when the selected image changes.
          """
        index = int(self.image_list.currentIndex().row())
        if index != -1:
            if index >= len(self.image_data_list) - 1:
                index = len(self.image_data_list) - 1
            self.current_image = self.image_data_list[index][0]
            self.update_color_pallet(self.image_data_list[index][1])
            self.update_preview_image(self.current_image)

    def delete_image(self, index):
        """
        Delete an image from the application.

        Parameters:
        index (int): The index of the image to delete.
        """
        del self.image_data_list[index]
        if not len(self.image_data_list):
            self.update_preview_image('', 'reset')
            self.color_display.reset_layout()

    def get_image(self):
        """
        Get the currently selected image from the list.

        Returns:
        np.array: The image read and converted to RGB.
        """
        item = self.image_list.item(self.image_list.count() - 1)
        image_path = item.value

        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def extract_colors(self, image):
        """
        Extract the dominant colors from an image.

        Parameters:
        image (np.array): The input image.

        Returns:
        list: A sorted list of dominant colors.
        """
        unique_colors = np.unique(image, axis=0)
        image_2d = image.reshape((image.shape[0] * image.shape[1], 3))
        clusters_num = 5 + (len(unique_colors) // 6)
        clusters_num = 5 if clusters_num < 2 else 24 if clusters_num > 24 else clusters_num
        clt = KMeans(n_clusters=clusters_num)
        clt.fit(image_2d)

        hist = self.centerize_histogram(clt)

        color_histogram = [[int(c) for c in color[1]] for color in list(zip(hist, clt.cluster_centers_))]

        return self.sort_colors(color_histogram)

    def centerize_histogram(self, clt):
        """
        Compute and normalize the histogram of cluster centers.

        Parameters:
        clt (KMeans): The KMeans clustering model.

        Returns:
        np.array: The normalized histogram.
        """
        num_labels = np.arange(0, len(np.unique(clt.labels_)) + 1)
        hist, _ = np.histogram(clt.labels_, bins=num_labels)
        hist = hist.astype("float")
        hist /= hist.sum()
        return hist

    def sort_colors(self, color_list):
        """
        Sort colors first by hue and then by luminance.

        Parameters:
        color_list (list): A list of colors to sort.

        Returns:
        list: The sorted list of colors.
        """
        def luminance(color):
            return 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]

        reds = []
        greens = []
        blues = []
        rests = []
        for color in color_list:
            max_color = max(*color)
            if color[0] == max_color:
                reds.append(color)
            elif color[1] == max_color:
                greens.append(color)
            elif color[2] == max_color:
                blues.append(color)
            else:
                rests.append(color)
        reds = sorted(reds, key=luminance)
        greens = sorted(greens, key=luminance)
        blues = sorted(blues, key=luminance)
        rests = sorted(rests, key=luminance)

        return [*reds, *greens, *blues, *rests]

    def update_color_pallet(self, color_list):
        """
        Update the color palette display with the given colors.

        Parameters:
        color_list (list): The list of colors to display.
        """
        self.color_display.reset_layout()
        row, col = 0, 0

        for idx, color in enumerate(color_list):
            shape = GUI.Shape(self.color_display, 'circle', str(color), name=f'color_{idx}', fill_color=color,
                      grid_location=(row, col), action=self.copy_color)
            shape.custom_functions['mouse_release'] = lambda color = shape.shape.fill_color,widget = shape: self.copy_color(color,widget)
            shape.custom_functions['mouse_leave'] = lambda widget = shape: self.reset_copy_message(widget)
            col += 1
            if col == 6:
                col = 0
                row += 1

    def update_preview_image(self, image, mode=None):
        """
        Update the image preview display.

        Parameters:
        image (QPixmap): The image to display.
        mode (str): The mode of updating ('reset' to clear the display).
        """
        if mode == 'reset':
            self.preview_image.clear()
        else:
            self.preview_image.set_image_background(pixmap=image)

    def copy_color(self, color, widget):
        """
        Copy the selected color to the clipboard.

        Parameters:
        color (str): The color to copy.
        widget (QWidget): The widget displaying the color.
        """
        if 'copied.' not in widget.toolTip():
            widget.setToolTip(f'{widget.toolTip()} copied.')

        pos = widget.mapToGlobal(widget.rect().center())
        GUI.QToolTip.showText(pos, widget.toolTip())
        pyperclip.copy(color)

    def reset_copy_message(self,widget):
        if 'copied' in widget.toolTip():
            widget.setToolTip(f'{widget.toolTip()[:-7]}')

    def cv2qt_image(self, image):
        """
        Convert a cv2 image to a QPixmap.

        Parameters:
        image (np.array): The input image.

        Returns:
        QPixmap: The converted QPixmap.
        """
        height, width, channel = image.shape

        q_image = GUI.QImage(image.data, width, height, width * channel, GUI.QImage.Format_RGB888)
        return GUI.QPixmap.fromImage(q_image)


if __name__ == '__main__':
    ch = ColorHunter(700, 520)
    ch.main_window.run()
