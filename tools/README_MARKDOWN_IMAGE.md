# Markdown 配图生成器 - 快速开始指南

## 🚀 5 分钟快速上手

### 步骤 1: 安装依赖 (1 分钟)

```bash
pip install zai-sdk requests
```

### 步骤 2: 设置 API Key (1 分钟)

**获取 API Key:**
1. 访问 [智谱 AI 开放平台](https://open.bigmodel.cn/)
2. 注册/登录账号
3. 在控制台创建 API Key

**设置环境变量:**

Windows PowerShell:
```powershell
$env:ZHIPU_API_KEY='sk-你的 APIKey'
```

Linux/Mac:
```bash
export ZHIPU_API_KEY='sk-你的 APIKey'
```

### 步骤 3: 运行测试 (1 分钟)

```bash
python tools/test_installation.py
```

如果所有测试通过，继续下一步。

### 步骤 4: 准备你的 Markdown 文件 (1 分钟)

在你的 Markdown 文件中添加配图标记:

**方法 1 - 占位符语法:**
```markdown
![AI 神经网络架构示意图，展示数据流动和层次结构](placeholder)
```

**方法 2 - HTML 注释:**
```markdown
<!-- 这里需要一张图：卷积神经网络 CNN 结构示意图 -->
```

**方法 3 - 引用语法:**
```markdown
[//]: # (image: 微服务架构图，展示服务间的调用关系)
```

### 步骤 5: 生成图片 (1 分钟)

```bash
python tools/md_image_generator.py your_article.md
```

完成！工具会自动:
1. 识别文中的配图需求
2. 调用 GLM-Image 生成图片
3. 下载图片到 `images/` 目录
4. 更新 Markdown 文件的图片引用

## 📝 完整示例

### 示例文章

创建 `test_article.md`:

```markdown
# 深度学习入门教程

## 什么是神经网络？

神经网络是一种模仿生物大脑结构的计算模型。

![三层神经网络示意图：输入层、隐藏层和输出层，展示神经元之间的连接](placeholder)

如上图所示，神经网络包含多个层次的神经元...

## 卷积神经网络

<!-- 这里需要一张图：CNN 的卷积操作示意图，展示滤波器如何在图像上滑动 -->

卷积神经网络 (CNN) 通过卷积核提取图像特征。

[//]: # (image: 池化层工作原理图，展示最大池化和平均池化的区别)

池化层用于降低特征图的维度...
```

### 执行命令

```bash
python tools/md_image_generator.py test_article.md -o article_images
```

### 查看结果

生成的目录结构:
```
project/
├── test_article.md          # 已自动更新图片路径
├── article_images/           # 新建的图片目录
│   ├── 20260303_120000_a1b2c3d4_三层神经网络示意图.png
│   ├── 20260303_120010_e5f6g7h8_CNN 的卷积操作示意图.png
│   └── 20260303_120020_i9j0k1l2_池化层工作原理图.png
└── tools/
    └── md_image_generator.py
```

Markdown 文件变化:
```markdown
# 从这样:
![三层神经网络示意图...](placeholder)

# 变成这样:
![三层神经网络示意图...](article_images/20260303_120000_a1b2c3d4_三层神经网络示意图.png)
```

## 🔧 常用命令

### 处理单个文件
```bash
python tools/md_image_generator.py article.md
```

### 批量处理整个目录
```bash
python tools/md_image_generator.py --dir docs/
```

### 仅预览 (不实际修改)
```bash
python tools/md_image_generator.py article.md --dry-run
```

### 指定图片尺寸
```bash
python tools/md_image_generator.py article.md --size 1728x960
```

### 自定义输出目录
```bash
python tools/md_image_generator.py article.md -o assets/images
```

## 💡 撰写优质描述的技巧

好的描述 = 高质量的图片

### ✅ 推荐写法

```markdown
![一只橘色的波斯猫坐在阳光充足的窗台上，窗外是蓝天白云，温暖的光线洒在猫咪身上]
```

### ❌ 避免简略写法

```markdown
![一只猫](placeholder)
```

### 描述要素

1. **主体**: 清晰描述主要对象
2. **场景**: 说明背景和环境
3. **风格**: 指定艺术风格或视觉效果
4. **细节**: 添加颜色、光线、材质等细节

## 🎨 支持的图片尺寸

GLM-Image 支持多种尺寸 (单位：像素):

| 尺寸 | 比例 | 适用场景 |
|------|------|----------|
| 1280×1280 | 1:1 | 通用示意图 |
| 1568×1056 | 3:4 | 横向图表 |
| 1056×1568 | 4:3 | 纵向图表 |
| 1728×960 | 16:9 | 宽屏展示 |
| 960×1728 | 9:16 | 手机竖屏 |

**使用示例:**
```bash
python tools/md_image_generator.py article.md --size 1728x960
```

## ⚠️ 常见问题

### Q1: 提示 API Key 错误

**解决:**
```powershell
# 检查是否设置成功
echo $env:ZHIPU_API_KEY

# 重新设置
$env:ZHIPU_API_KEY='sk-正确的 APIKey'
```

### Q2: 依赖未安装

**解决:**
```bash
pip install zai-sdk requests
```

### Q3: 网络错误

**可能原因:**
- 网络连接问题
- 需要配置代理

**解决:**
```bash
# 检查网络
ping open.bigmodel.cn

# 如有需要，配置代理
set HTTP_PROXY=http://proxy-server:port
```

### Q4: 图片生成失败

**可能原因:**
- 描述包含敏感词
- 描述超过 1000 字符

**解决:**
- 简化或修改描述
- 查看详细错误信息

## 💰 费用说明

GLM-Image 收费标准：**0.1 元/次**

**成本优化建议:**
1. 先用 `--dry-run` 预览
2. 小批量测试后再批量生成
3. 复用高质量图片

## 📚 更多资源

- **完整使用文档**: [docs/markdown_image_generator_usage.md](docs/markdown_image_generator_usage.md)
- **技能说明**: [.qoder/skills/markdown-image-generator/SKILL.md](.qoder/skills/markdown-image-generator/SKILL.md)
- **GLM-Image 官方文档**: [docs/LLM_SDK/GLM-Image.md](docs/LLM_SDK/GLM-Image.md)
- **智谱 AI 平台**: https://open.bigmodel.cn/

## 🎯 下一步

1. ✅ 完成环境配置和测试
2. ✅ 阅读完整使用文档
3. ✅ 开始为你的文章生成精美配图!

祝你使用愉快！🚀
