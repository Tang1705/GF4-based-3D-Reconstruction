import cocos
from PIL import Image

from reconstruction import reconstruction
from cocos import layer


# noinspection PyGlobalUndefined
class MainScene(cocos.layer.Layer):
    def __init__(self):
        super(MainScene, self).__init__()
        d_width, d_height = cocos.director.director.get_window_size()

        background = cocos.sprite.Sprite('Resource//main_scene//background.png')
        background.position = d_width // 2, d_height // 2

        global origin, featurepoint, white, thidcoord

        origin_label = cocos.text.Label("相机画面", color=(0, 0, 0, 255))
        origin_label.position = 250, 680
        origin = cocos.sprite.Sprite('Resource//main_scene//display_panel.png')
        origin.position = 290, 530

        feature_point_label = cocos.text.Label("特征点", color=(0, 0, 0, 255))
        feature_point_label.position = 250, 330
        featurepoint = cocos.sprite.Sprite('Resource//main_scene//display_panel.png')
        featurepoint.position = 290, 180

        white_label = cocos.text.Label("原始图像", color=(0, 0, 0, 255))
        white_label.position = 720, 680
        white = cocos.sprite.Sprite('Resource//main_scene//display_panel.png')
        white.position = 750, 530

        thidcoord_label = cocos.text.Label("三维坐标", color=(0, 0, 0, 255))
        thidcoord_label.position = 720, 330
        thidcoord = cocos.sprite.Sprite('Resource//main_scene//display_panel.png')
        thidcoord.position = 750, 180

        self.add(background)
        self.add(origin_label)
        self.add(origin)
        self.add(feature_point_label)
        self.add(featurepoint)
        self.add(white_label)
        self.add(white)
        self.add(thidcoord_label)
        self.add(thidcoord)

        main_menu = Main_Menu()
        self.add(main_menu)

    def camera(self):
        img = Image.open('square.png')
        image = img.resize((320, 256), Image.ANTIALIAS)
        image.save("Resource//main_scene//square_display.png")
        origin = cocos.sprite.Sprite('Resource//main_scene//square_display.png')
        origin.position = 290, 530
        self.add(origin)

        img = Image.open('square.png')
        image = img.resize((320, 256), Image.ANTIALIAS)
        image.save("Resource//main_scene//square_display.png")
        origin = cocos.sprite.Sprite('Resource//main_scene//square_display.png')
        origin.position = 750, 530
        self.add(origin)

    def feature_point_reconstruct(self):
        img = Image.open('feature_point.png')
        image = img.resize((320, 256), Image.ANTIALIAS)
        image.save("Resource//main_scene//feature_point.png")
        featurepoint = cocos.sprite.Sprite('Resource//main_scene//feature_point.png')
        featurepoint.position = 290, 180
        self.add(featurepoint)

        img = Image.open('Resource//main_scene//reconstruction_result.png')
        image = img.resize((320, 256), Image.ANTIALIAS)
        image.save("Resource//main_scene//reconstruction_result.png")
        thicoord = cocos.sprite.Sprite('Resource//main_scene//reconstruction_result.png')
        thicoord.position = 750, 180
        self.add(thicoord)


# 自定义菜单类

class Main_Menu(cocos.menu.Menu):

    def __init__(self):
        super(Main_Menu, self).__init__()
        cocos.layer.Layer.is_event_handler = True
        d_width, d_height = cocos.director.director.get_window_size()

        # 文本菜单项  （文字，回掉函数）

        item1 = cocos.menu.MenuItem('标定', self.item1_callback)

        item2 = cocos.menu.MenuItem('投影', self.item2_callback)

        item3 = cocos.menu.MenuItem('拍照', self.item3_callback)

        item4 = cocos.menu.MenuItem('重建', self.item4_callback)

        item5 = cocos.menu.MenuItem('帮助', self.item5_callback)

        # 创建菜单（添加项的列表，选中效果，未选中效果）

        self.create_menu([item1, item2, item3, item4, item5],
                         layout_strategy=cocos.menu.fixedPositionMenuLayout(
                             [(1050, 550), (1050, 450), (1050, 350), (1050, 250), (1050, 150)]),

                         selected_effect=cocos.menu.shake(),

                         unselected_effect=cocos.menu.shake_back(),

                         )
        # 改变字体

        self.font_item['font_size'] = 8

        # 选中时

        self.font_item_selected['font_size'] = 16

    def item1_callback(self):
        pass

    def item2_callback(self, value):
        pass

    def item3_callback(self):
        MainScene.camera(self.parent)

    def item4_callback(self):
        reconstruction()
        MainScene.feature_point_reconstruct(self.parent)

    def item5_callback(self):
        pass


def create():
    # 将背景层  添加到场景
    bg = MainScene()
    global main_scene
    main_scence = cocos.scene.Scene(bg)
    # main_menu = Main_Menu()
    # main_scence.add(main_menu)
    return main_scence
