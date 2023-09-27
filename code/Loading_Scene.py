import time

import cocos
import pyglet
from cocos import layer

# 背景层
from cocos.actions import MoveTo, FadeIn, Delay
from cocos.scenes import FadeTRTransition

import Main_Scene


class LoadingScene(cocos.layer.Layer):

    def __init__(self):
        super(LoadingScene, self).__init__()
        # 开启事件处理层 有事件发生 就要写
        cocos.layer.Layer.is_event_handler = True
        d_width, d_height = cocos.director.director.get_window_size()

        # 创建背景精灵
        background = cocos.sprite.Sprite('Resource//loading_scene//background.png')
        background.position = d_width // 2, d_height // 2

        # 创建标题精灵
        title = cocos.sprite.Sprite('Resource//loading_scene//title.png')
        title.position = d_width // 2, d_height // 2

        delay_move_to = Delay(1.0)
        move_to = MoveTo((d_width // 2, d_height // 5 * 3), 2)
        title.do(delay_move_to + move_to)

        sub_title = cocos.sprite.Sprite('Resource//loading_scene//sub_title.png')
        sub_title.position = d_width // 2, d_height // 7 * 3
        sub_title.opacity = 0
        fade_in = FadeIn(2.0)
        delay_fade_in = Delay(2)
        sub_title.do(delay_fade_in + fade_in)

        self.add(background)
        self.add(title)
        self.add(sub_title)

    def on_key_press(self, ke, modifiiers):  # 修改方法

        if ke == pyglet.window.key.SPACE or ke == pyglet.window.key.SPACE:  # 空格键
            set_sence = Main_Scene.create()
            donghua = FadeTRTransition(set_sence, 1.5)  # 导包！！
            cocos.director.director.push(donghua)


if __name__ == '__main__':
    # 初始化导演

    cocos.director.director.init(width=1200, height=750, caption="高铁轮轨三维姿态重建软件")

    # 将背景层  添加到场景

    bg = LoadingScene()

    main_scence = cocos.scene.Scene(bg)

    # 启动场景

    cocos.director.director.run(main_scence)
