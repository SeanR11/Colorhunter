from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from pathlib import Path
import os

class Window(QMainWindow):
    def __init__(self, width, height, title, background_color=(0, 0, 0), icon=None, resizeable=False, titlebar=True):
        """
        Initialize the Window class.

        Args:
            width (int): The width of the window.
            height (int): The height of the window.
            title (str): The title of the window.
            background_color (tuple): RGB values for the background color. Default is black (0, 0, 0).
            resizeable (bool): If False, the window cannot be resized. Default is False.
            titlebar (bool): If False, the window will be frameless. Default is True.
        """
        self.app = QApplication([])
        super().__init__()
        self.screen_size = QSize(self.app.desktop().width(), self.app.desktop().height())
        self.size = QSize(width, height)
        self.toolbar = None  # Toolbar will be initialized in enable_toolbar method
        self.central = QWidget()

        self.setWindowTitle(title)
        self.setCentralWidget(self.central)

        if not titlebar:
            self.setWindowFlags(Qt.FramelessWindowHint)
        if not resizeable:
            self.setFixedSize(self.size)
        if icon:
            self.setWindowIcon(QIcon(icon))

        # Center the window on the screen
        self.setGeometry((self.screen_size.width() // 2 - self.size.width() // 2),
                         (self.screen_size.height() // 2 - self.size.height() // 2),
                         self.size.width(), self.size.height())
        self.setStyleSheet(f'background:rgb{background_color}')  # Set background color

    def enable_toolbar(self, toolbar_widget, background=(255, 255, 255), area=Qt.TopToolBarArea, hideable=False,
                       movable=False):
        """
        Enable a toolbar for the window.

        Args:
            toolbar_widget (QWidget): The widget to be added to the toolbar.
            background (tuple): RGB values for the toolbar background color. Default is white (255, 255, 255).
            area (Qt.ToolBarArea): The area to place the toolbar. Default is Qt.TopToolBarArea.
            hideable (bool): If False, the toolbar cannot be hidden. Default is False.
            movable (bool): If True, the toolbar can be moved. Default is False.
        """
        self.toolbar = QToolBar()
        self.toolbar.setStyleSheet(f'background:rgb{background};padding:0px;margin:0px')
        self.toolbar.setMovable(movable)
        if not hideable:
            self.toolbar.setContextMenuPolicy(Qt.PreventContextMenu)
        self.toolbar.addWidget(toolbar_widget)
        self.addToolBar(area, self.toolbar)

    def run(self):
        """
        Show the window and start the application event loop.
        """
        self.show()
        self.app.exec_()


class Layout:
    def __init__(self, parent, name, layout_type, alignment=None, size: QSize = None, background_color=None,
                 grid_location=None):
        """
        Initialize the Layout class.

        Args:
            parent (QWidget or Layout): The parent widget or layout.
            name (str): The object name for the layout.
            layout_type (str): Type of layout ('v' for QVBoxLayout, 'h' for QHBoxLayout, 'g' for QGridLayout).
            alignment (str, optional): Alignment of the layout ('center', 'right', 'left', 'centerh', 'centerv', 'top', 'bot', 'centerbot', 'centertop').
            size (QSize, optional): Fixed size of the layout. Default is None.
            background_color (tuple, optional): RGB values for the background color. Default is None.
            grid_location (tuple, optional): Grid location if parent layout is QGridLayout. Default is None.
        """
        self.layout = self.get_layout(layout_type)
        self.parent = parent
        self.alignment = self.get_alignment(alignment)
        self.styles = {}
        self.grid_location = grid_location

        if self.alignment:
            self.layout.setAlignment(self.alignment)

        if isinstance(parent, QWidget):
            self.background_widget = parent
            self.background_widget.setLayout(self.layout)
        else:
            self.background_widget = QWidget()
            self.background_widget.setLayout(self.layout)
            if parent is not None:
                if self.grid_location:
                    parent.layout.addWidget(self.background_widget, self.grid_location[0], self.grid_location[1])
                else:
                    parent.layout.addWidget(self.background_widget)

        if size:
            if size.width() != 0:
                self.background_widget.setFixedWidth(size.width())
            if size.height() != 0:
                self.background_widget.setFixedHeight(size.height())
        if background_color:
            self.set_background(background_color)
        self.layout.setContentsMargins(2, 2, 2, 2)
        self.background_widget.setObjectName(name)

    def get_layout(self, layout_type):
        """
        Return the appropriate QLayout based on layout_type.

        Args:
            layout_type (str): The type of layout ('v', 'h', 'g').

        Returns:
            QLayout: The corresponding QLayout object.

        Raises:
            ValueError: If an invalid layout type is provided.
        """
        if layout_type == 'v':
            return QVBoxLayout()
        elif layout_type == 'h':
            return QHBoxLayout()
        elif layout_type == 'g':
            return QGridLayout()
        else:
            raise ValueError(
                "Invalid layout type. Use 'v' for QVBoxLayout, 'h' for QHBoxLayout, or 'g' for QGridLayout.")

    def get_alignment(self, alignment):
        """
        Return the appropriate Qt alignment based on alignment string.

        Args:
            alignment (str): The alignment type.

        Returns:
            Qt.Alignment: The corresponding Qt alignment.
        """
        total_align = 0
        if alignment:
            if 'center' in alignment:
                total_align |= Qt.AlignCenter
            if 'top' in alignment:
                total_align |= Qt.AlignTop
            if 'bot' in alignment:
                total_align |= Qt.AlignBottom
            if 'right' in alignment:
                total_align |= Qt.AlignRight
            if 'left' in alignment:
                total_align |= Qt.AlignLeft
            return total_align
        else:
            return Qt.AlignJustify

    def reset_layout(self):
        """
        Remove all widgets from the layout.
        """
        if self.layout is not None:
            while self.layout.count():
                item = self.layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

    def update_style(self):
        """
        Update the stylesheet of the background widget based on the styles dictionary.
        """
        styles = [f'{key}:{value};' for key, value in self.styles.items()]
        self.background_widget.setStyleSheet(f'#{self.background_widget.objectName()} {{ {"".join(styles)} }}')

    def enable_border(self, border_width, border_color):
        """
        Enable a border around the layout.

        Args:
            border_width (int): The width of the border.
            border_color (str): The color of the border.
        """
        self.styles['border'] = f"{border_width}px solid {border_color}"
        self.update_style()

    def set_background(self, background_color):
        """
        Set the background color of the layout.

        Args:
            background_color (tuple): RGB values for the background color.
        """
        self.styles['background-color'] = f'rgb({background_color[0]}, {background_color[1]}, {background_color[2]})'
        self.update_style()

    def addWidget(self, widget, grid_layout=None):
        """
        Add a widget to the layout.

        Args:
            widget (QWidget): The widget to be added.
            grid_layout (tuple, optional): The grid location if the layout is QGridLayout. Default is None.
        """
        if grid_layout:
            self.layout.addWidget(widget, grid_layout[0], grid_layout[1])
        elif self.grid_location:
            self.parent.layout.addWidget(widget, self.grid_location[0], self.grid_location[1])
        else:
            self.layout.addWidget(widget)

    def addLayout(self, layout):
        """
        Add another layout to this layout.

        Args:
            layout (Layout): The layout to be added.
        """
        self.layout.addLayout(layout.layout)

    def share_equal_space(self):
        """
        Set equal spacing for all widgets in the layout.
        """
        inner_layouts_num = self.layout.count()
        for i in range(inner_layouts_num):
            self.layout.setStretch(i, 1)


class Widget(QWidget):
    """
    Base widget class that all other widget components will inherit from.
    """

    def __init__(self, container_layout=None, name=None, size: QSize = None, font_size=None, font=None, color=None,
                 grid_location=None):
        """
        Initializes the Widget class.

        Args:
            container_layout: The layout container for the widget.
            name (str): The name of the widget.
            size (QSize): The size of the widget.
            font_size (int): The font size of the widget text.
            font (str): The font of the widget text.
            color (tuple): The color of the widget background.
            grid_location (tuple): The location of the widget in a grid layout.
        """
        super().__init__()
        self.layout = container_layout
        self.name = name
        self.font = font if font is not None else 'Arial'
        self.font_size = font_size if font_size is not None else 12
        self.size: QSize = size if size is not None else QSize(40, 40)
        self.color = color if color is not None else (255, 255, 255)
        self.styles = {}
        self.setObjectName(self.name)
        if grid_location:
            self.grid_location = grid_location
        if self.name == 'rusk':
            print('s')

    def show(self):
        """ Adds the widget to the layout and displays it. """
        if type(self.layout.layout) == QGridLayout:
            self.layout.addWidget(self, self.grid_location)
        else:
            self.layout.addWidget(self)

    def check_path(self, path, type):
        """
        Checks if the given path is valid and matches the specified type.

        Args:
            path (str): The file or folder path.
            type (str): The type to check ('image', 'pdf', 'audio', or 'folder').

        Returns:
            Path or bool: The valid path object if valid, False otherwise.
        """
        temp_path = Path(path)

        if temp_path.exists() and temp_path.is_file():
            if type == 'image':
                if temp_path.suffix in ['.jpg', '.png', '.jpeg']:
                    return temp_path
            if type == 'pdf':
                if temp_path.suffix in ['.pdf']:
                    return temp_path
            if type == 'audio':
                if temp_path.suffix in ['.ogg', '.wav', '.mp3', '.raw']:
                    return temp_path
        elif type == 'folder' and temp_path.exists():
            return temp_path
        else:
            print("Incorrect file path")
            return False

    def set_size(self, size=None):
        """
        Sets the size of the widget.

        Args:
            size (QSize, optional): The size to set for the widget. Defaults to None.
        """
        if size is not None:
            self.size = size
        if self.size.width() != 0:
            self.setFixedWidth(self.size.width())
        if self.size.height() != 0:
            self.setFixedHeight(self.size.height())

    def update_style(self):
        """ Updates the stylesheet of the widget based on the current styles. """
        styles = ""
        for pseudo in self.styles:
            if pseudo == 'none':
                styles += f"#{self.objectName()} {{" + "".join(
                    f'{key}:{value}; ' for key, value in self.styles[pseudo].items()) + "}"
            else:
                styles += f"#{self.objectName()}::{pseudo} {{" + "".join(
                    f'{key}:{value};' for key, value in self.styles[pseudo].items()) + "}"

        self.setStyleSheet(styles)

    def add_style(self, pseudo_element, style_property, style_value):
        """
        Adds a style to the widget.

        Args:
            pseudo_element (str): The pseudo element for the style.
            style_property (str): The CSS style property.
            style_value (str): The value of the CSS style property.
        """
        if pseudo_element not in self.styles.keys():
            self.styles[pseudo_element] = {}
        self.styles[pseudo_element][style_property] = style_value
        self.update_style()

    def get_alignment(self, alignment):
        """
        Returns the Qt alignment constant based on the alignment string.

        Args:
            alignment (str): The alignment string ('center', 'right', 'left', etc.)

        Returns:
            Qt.Alignment: The corresponding Qt alignment constant.
        """
        total_align = 0
        if alignment:
            if 'center' in alignment:
                total_align |= Qt.AlignCenter
            if 'top' in alignment:
                total_align |= Qt.AlignTop
            if 'bot' in alignment:
                total_align |= Qt.AlignBottom
            if 'right' in alignment:
                total_align |= Qt.AlignRight
            if 'left' in alignment:
                total_align |= Qt.AlignLeft
            return total_align
        else:
            return Qt.AlignJustify


class Button(QPushButton, Widget):
    """
    Button widget class inheriting from QPushButton and Widget.
    """

    def __init__(self, container_layout, name=None, text=None, icon=None, font_size=None, size: QSize = None,
                 action=None):
        """
        Initializes the Button class.

        Args:
            container_layout (Layout): The layout container for the button.
            text (str, optional): The text displayed on the button. Defaults to None.
            icon (str, optional): The path to the icon image for the button. Defaults to None.
            font_size (int, optional): The font size of the button text. Defaults to None.
            size (QSize, optional): The size of the button. Defaults to None.
            action (callable, optional): The action to be triggered on button click. Defaults to None.
        """
        QPushButton.__init__(self)
        Widget.__init__(self, container_layout=container_layout, name=name, size=size, font_size=font_size)

        if icon:
            icon_path = self.check_path(icon, 'image')
            if icon_path:
                self.setIcon(QIcon(str(icon_path)))
                self.setIconSize(self.size)
        elif text:
            self.setText(text)
            self.setFont(QFont(self.font, self.font_size))
        if action:
            self.clicked.connect(action)

        self.add_style('none', 'border', 'none')
        self.setFixedSize(self.size)
        self.show()

    def update_image(self, file_path, size: QSize = None):
        """
        Updates the button's icon image.

        Args:
            file_path (str): The path to the new icon image.
            size (QSize, optional): The size of the new icon. Defaults to None.
        """
        image_path = self.check_path(file_path, 'image')
        if image_path:
            self.setIcon(QIcon(str(image_path)))
            if size:
                self.setIconSize(size)
                self.setFixedSize(size)


class Label(QLabel, Widget):
    """
    Label widget class inheriting from QLabel and Widget.
    """

    def __init__(self, container_layout, text, name=None, size: QSize = None, alignment='center', font_size=None,
                 font=None, color=None, grid_location=None):
        """
        Initializes the Label class.

        Args:
            container_layout (Layout): The layout container for the label.
            text (str): The text displayed on the label.
            name (str, optional): The name identifier for the label. Defaults to None.
            size (QSize, optional): The size of the label. Defaults to None.
            alignment (str, optional): The alignment of the label text. Defaults to 'center'.
            font_size (int, optional): The font size of the label text. Defaults to None.
            font (str, optional): The font family of the label text. Defaults to None.
            color (tuple, optional): The RGB color of the label text. Defaults to None.
            grid_location (tuple, optional): The location in the grid if the layout is a QGridLayout. Defaults to None.
        """
        Widget.__init__(self, container_layout=container_layout, size=size, font_size=font_size, font=font, color=color,
                        grid_location=grid_location)
        QLabel.__init__(self, text)
        self.alignment = self.get_alignment(alignment)

        if name:
            self.setObjectName(name)
        if size:
            self.set_size(size)
        self.setFont(QFont(self.font, self.font_size))
        if color:
            self.add_style('none', 'color', f'rgb{color}')
        self.setAlignment(self.alignment)
        self.show()

    def add_style(self, pseudo_element, style_property, style_value):
        """
        Adds a style to the label.

        Args:
            pseudo_element (str): The pseudo-element to which the style applies.
            style_property (str): The CSS style property.
            style_value (str): The value of the CSS style property.
        """
        super().add_style(pseudo_element, style_property, style_value)

    def set_image_background(self, pixmap=None, image_path=None, new_size=None):
        """
        Sets an image as the background of the label.

        Args:
            pixmap (QPixmap, optional): The pixmap image to set as background. Defaults to None.
            image_path (str, optional): The path to the image file to set as background. Defaults to None.
            new_size (QSize, optional): The new size of the label. Defaults to None.
        """
        if image_path or pixmap:
            if image_path:
                pixmap = QPixmap(image_path)
            if new_size:
                self.set_size(new_size)
            self.setPixmap(pixmap.scaled(self.width(), self.height(), Qt.KeepAspectRatio))


class ToolButton(QToolButton, Widget):
    """
    ToolButton widget class inheriting from QToolButton and Widget.
    """

    def __init__(self, container_layout, text, action=None, tooltip=None, font=None, font_size=None, size: QSize = None,
                 color=None):
        """
        Initializes the ToolButton class.

        Args:
            container_layout (Layout): The layout container for the button.
            text (str): The text displayed on the button.
            action (function, optional): The function to be executed when the button is clicked. Defaults to None.
            tooltip (str, optional): The tooltip text displayed when hovering over the button. Defaults to None.
            font (str, optional): The font family of the button text. Defaults to None.
            font_size (int, optional): The font size of the button text. Defaults to None.
            size (QSize, optional): The size of the button. Defaults to None.
            color (tuple, optional): The RGB color of the button text. Defaults to None.
        """
        super().__init__(container_layout=container_layout, font=font, font_size=font_size, size=size, color=color)
        QToolButton.__init__(self)
        self.setText(text)
        self.setFont(QFont(self.font, self.font_size))
        self.setStyleSheet(f'border:none;color:rgb{color}')
        if tooltip:
            self.setToolTip(tooltip)
        if action:
            self.clicked.connect(action)

        self.show()


class ScaleBar(QProgressBar, Widget):
    """
    ScaleBar widget class inheriting from QProgressBar and Widget.
    """

    def __init__(self, container_layout, name=None, action=None, size: QSize = None, color=None, default_text=False):
        """
        Initializes the ScaleBar class.

        Args:
            container_layout (Layout): The layout container for the scale bar.
            name (str, optional): The name identifier for the scale bar. Defaults to None.
            action (function, optional): The function to be executed on mouse press event. Defaults to None.
            size (QSize, optional): The size of the scale bar. Defaults to None.
            color (tuple, optional): The RGB color of the scale bar. Defaults to None.
            default_text (bool, optional): Flag indicating whether to show default text on the scale bar. Defaults to False.
        """
        super().__init__(container_layout=container_layout, size=size, color=color)
        QProgressBar.__init__(self)
        self.progress = 0
        self.action = action
        if name:
            self.setObjectName(name)
        if size:
            if size.width() != 0:
                self.setFixedWidth(size.width())
            if size.height() != 0:
                self.setFixedHeight(size.height())
        if not default_text:
            self.setTextVisible(False)
        if color:
            self.add_style('chunk', 'background-color', color)

        self.setRange(0, 100)
        self.show()

    def mousePressEvent(self, event):
        """
        Overrides the default mouse press event to execute custom action.

        Args:
            event (QMouseEvent): The mouse press event object.
        """
        if event.button() == Qt.LeftButton:
            self.action(event)


class List(QListWidget, Widget):
    """
    List widget class inheriting from QListWidget and Widget.
    """

    def __init__(self, container_layout, name=None, size: QSize = None, action=None, color=None, side='left'):
        """
        Initializes the List class.

        Args:
            container_layout (Layout): The layout container for the list.
            name (str, optional): The name identifier for the list. Defaults to None.
            size (QSize, optional): The size of the list. Defaults to None.
            action (function, optional): The function to be executed on list item change. Defaults to None.
            color (tuple, optional): The RGB color of the list background. Defaults to None.
            side (str, optional): The side on which buttons are aligned ('left' or 'right'). Defaults to 'left'.
        """
        Widget.__init__(self, container_layout=container_layout, size=size)
        QListWidget.__init__(self)

        self.inner_layout = Layout(container_layout, f'{self.objectName()}_inner_layout', 'v', size=size)
        self.add_item_function = None
        self.change_item_function = None
        self.delete_item_function = None
        self.custom_functions = {}

        if name:
            self.setObjectName(name)
        if color:
            self.add_style('none', 'background-color', color)
        if size:
            self.setFixedSize(size)

        self.load_buttons(side)
        self.inner_layout.addWidget(self)

    def load_buttons(self, side):
        """
        Loads buttons for adding and deleting items to/from the list.

        Args:
            side (str): The side on which buttons are aligned ('left' or 'right').
        """
        self.buttons_layout = Layout(self.inner_layout, f'{self.objectName()}_button_layout', 'h', size=QSize(0, 30))
        self.buttons_layout.layout.setAlignment(
            self.get_alignment(alignment=side) | self.get_alignment(alignment='bot'))
        self.add_button = Button(self.buttons_layout, name=f"{self.objectName()}_add_button",
                                 icon='assets/img/add_button.png', size=QSize(20, 20),
                                 action=self.add_item)
        self.delete_button = Button(self.buttons_layout, name=f'{self.objectName()}_del_button',
                                    icon='assets/img/delete_button.png', size=QSize(20, 20),
                                    action=self.delete_item)

    def add_item(self):
        """
        Opens a file dialog to add a new item to the list.
        """
        file_name, _ = QFileDialog.getOpenFileName(self, 'Add Item', filter=self.item_filter)
        if file_name != '':
            new_item = ListItem(self, file_name.split('/')[-1], file_name)
            if 'add_item' in self.custom_functions:
                self.custom_functions['add_item']()

    def delete_item(self):
        """
        Deletes the selected item from the list.
        """
        index = int(self.currentRow())
        if index != -1:
            self.takeItem(index)

            if 'del_item' in self.custom_functions:
                self.custom_functions['del_item'](index)

    def set_custom_function(self, func_name, func):
        """
        Sets a custom function to be executed for a specific action.

        Args:
            func_name (str): The name of the function to set (e.g., 'add_item', 'del_item', 'item_changed').
            func (function): The function to be executed.
        """
        self.custom_functions[func_name] = func

    def currentChanged(self, *args, **kwargs):
        """
        Executes the custom function when the current item in the list changes.
        """
        if 'item_changed' in self.custom_functions:
            self.custom_functions['item_changed']()

    def set_item_filter(self, filter=None):
        """
        Sets the filter for allowed file types when adding items to the list.

        Args:
            filter (str, optional): The filter type ('img' for images). Defaults to None.
        """
        if filter == 'img':
            self.item_filter = "Images (*.jpg *.png *.jpeg)"


class ListItem(QListWidgetItem):
    """
    ListItem widget class inheriting from QListWidgetItem and Widget.
    """

    def __init__(self, parent_list, name, value):
        """
        Initializes the ListItem class.

        Args:
            parent_list (List): The parent List widget to which this item belongs.
            name (str): The name or display text of the list item.
            value (str): The value or associated data of the list item.
        """

        super().__init__()

        self.parent_list = parent_list
        self.value = value
        self.setText(name)
        self.parent_list.addItem(self)


class ShapeWidget(Widget):
    """
    Widget representing a shape with customizable fill and outline colors.
    """

    def __init__(self, container_layout, type, fill_color=(0, 0, 0), outline_color=(255, 255, 255)):
        """
        Initializes the ShapeWidget.

        Args:
            container_layout (Layout): The layout container for the widget.
            type (str): Type of the shape.
            fill_color (tuple, optional): RGB tuple for the fill color. Defaults to (0, 0, 0) (black).
            outline_color (tuple, optional): RGB tuple for the outline color. Defaults to (255, 255, 255) (white).
        """
        super().__init__(container_layout)
        self.type = type
        self.fill_color = fill_color
        self.outline_color = outline_color

        self.setAutoFillBackground(True)
        self.setAttribute(Qt.WA_StyledBackground, True)

        self.set_size()

    def paintEvent(self, event):
        """
        Handles the painting of the shape widget.

        Args:
            event (QPaintEvent): The paint event.
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        pen = QPen(QColor(*self.outline_color), 2, Qt.SolidLine)
        brush = QBrush(QColor(*self.fill_color), Qt.SolidPattern)

        painter.setPen(pen)
        painter.setBrush(brush)

        rect = self.rect().adjusted(2, 2, -2, -2)
        painter.drawEllipse(rect)


class Shape(Widget):
    """
    Widget representing a shape with customizable appearance and actions.
    """

    def __init__(self, container_layout, type, tooltip=None, name=None, size: QSize = None, action=None, color=None,
                 fill_color=(0, 0, 0), outline_color=(255, 255, 255), grid_location=(0, 0)):
        """
        Initializes the Shape.

        Args:
            container_layout (Layout): The layout container for the widget.
            type (str): Type of the shape.
            tooltip (str, optional): Tooltip text for the shape. Defaults to None.
            name (str, optional): Name identifier for the shape widget. Defaults to None.
            size (QSize, optional): Size of the shape widget. Defaults to None.
            action (function, optional): Function to be executed on mouse release. Defaults to None.
            color (tuple, optional): RGB tuple for the background color. Defaults to None.
            fill_color (tuple, optional): RGB tuple for the fill color of the shape. Defaults to (0, 0, 0) (black).
            outline_color (tuple, optional): RGB tuple for the outline color of the shape. Defaults to (255, 255, 255) (white).
            grid_location (tuple, optional): Grid location of the shape in the layout. Defaults to (0, 0).
        """
        super().__init__(container_layout, name=name, size=size, color=fill_color)
        self.inner_layout = Layout(self, f'{name}_inner_layout', 'v', size=QSize(0, 60), alignment='center')
        self.shape = ShapeWidget(self.inner_layout, type, fill_color, outline_color)
        self.custom_functions = {}
        self.custom_functions['mouse_release'] = action

        if tooltip:
            self.setToolTip(tooltip)
        self.inner_layout.addWidget(self.shape)
        self.layout.addWidget(self, grid_location)

    def mouseReleaseEvent(self, event):
        """
        Handles the mouse release event.

        Args:
            event (QMouseEvent): The mouse event.
        """
        if 'mouse_release' in self.custom_functions:
            self.custom_functions['mouse_release']()

    def leaveEvent(self, event):
        """
        Handles the leave event.

        Args:
            event (QEvent): The event.
        """
        if 'mouse_leave' in self.custom_functions:
            self.custom_functions['mouse_leave']()

    def change_color(self, outline_color=None, fill_color=None):
        """
        Changes the colors of the shape.

        Args:
            outline_color (tuple, optional): RGB tuple for the new outline color. Defaults to None.
            fill_color (tuple, optional): RGB tuple for the new fill color. Defaults to None.
        """
        if outline_color:
            self.outline_colorc = outline_color
        if fill_color:
            self.fill_color = fill_color

    def set_text(self, text, font_size=None, font=None, color=(0, 0, 0)):
        """
        Sets text on the shape.

        Args:
            text (str): The text to set.
            font_size (int, optional): The font size of the text. Defaults to None.
            font (str, optional): The font of the text. Defaults to None.
            color (tuple, optional): RGB tuple for the text color. Defaults to (0, 0, 0) (black).
        """
        if font:
            self.font = font
        if font_size:
            self.font_size = font_size
        self.text = Label(self.inner_layout, text, name=f'{self.objectName()}_text', font_size=font_size, font=font,
                          color=color)
        self.text.setAlignment(Qt.AlignCenter)

    def set_custom_function(self, func_name, func):
        """
        Sets a custom function to be executed for a specific action.

        Args:
            func_name (str): The name of the function to set (e.g., 'add_item', 'del_item', 'item_changed').
            func (function): The function to be executed.
        """
        self.custom_functions[func_name] = func
