#!/usr/bin/env python3
"""
批量生成 Git 入门专栏配图
"""
import os
import requests
from pathlib import Path
from datetime import datetime

try:
    from zai import ZhipuAiClient
except ImportError:
    print("请先安装: pip install zai-sdk")
    exit(1)

# 从 .env 加载 API Key
from dotenv import load_dotenv
load_dotenv()

# 图片描述列表
IMAGES = [
    # 第1篇
    {
        "filename": "version_control_evolution.png",
        "prompt": "技术示意图：版本控制演进图，展示从手动备份文件、本地版本控制系统（VCS）到分布式版本控制（Git）的发展历程，简洁的流程图风格，使用蓝色和绿色配色"
    },
    {
        "filename": "centralized_vcs.png",
        "prompt": "技术架构图：集中式版本控制系统架构，展示中央服务器连接多个开发者客户端的星型拓扑结构，服务器在中心，客户端围绕四周，专业简洁风格"
    },
    {
        "filename": "distributed_vcs.png",
        "prompt": "技术架构图：分布式版本控制系统架构，展示多个开发者各自拥有完整仓库副本的对等网络结构，每个节点都包含完整的版本历史，蓝色科技风格"
    },
    
    # 第2篇
    {
        "filename": "git_repository_structure.png",
        "prompt": "技术示意图：Git仓库目录结构，展示.git隐藏文件夹、工作区文件、配置文件等组成，树状结构图，清晰标注各部分功能"
    },
    {
        "filename": "git_three_areas.png",
        "prompt": "技术流程图：Git三个工作区域关系图，展示工作区（Working Directory）、暂存区（Staging Area）、本地仓库（Local Repository）之间的数据流动，使用箭头标注add、commit等命令，简洁专业风格"
    },
    
    # 第3篇
    {
        "filename": "github_features.png",
        "prompt": "技术架构图：GitHub核心功能架构，展示代码托管、协作开发、项目管理、持续集成、开源社区五大模块的组成，圆形放射状布局，配色柔和"
    },
    {
        "filename": "create_github_repo.png",
        "prompt": "界面示意图：GitHub创建新仓库界面，展示填写仓库名称、描述、选择公开/私有等关键表单元素的布局，模拟真实界面风格"
    },
    {
        "filename": "git_push_flow.png",
        "prompt": "技术流程图：Git push推送流程，展示从工作区到暂存区到本地仓库再到远程仓库的完整数据流，标注add、commit、push命令，箭头清晰，步骤明确"
    },
    
    # 第4篇
    {
        "filename": "git_branches.png",
        "prompt": "技术示意图：Git分支结构，展示主分支（main）和功能分支（feature）的分叉与合并，时间线从左到右，分支用不同颜色区分，节点标注commit"
    },
    {
        "filename": "fast_forward_merge.png",
        "prompt": "技术流程图：Fast-forward快进合并示意图，展示合并前后分支指针的移动过程，线性结构，箭头标注合并方向，简洁清晰"
    },
    {
        "filename": "three_way_merge.png",
        "prompt": "技术流程图：三方合并示意图，展示两个分支从分叉点分别开发后合并创建新merge commit的过程，标注共同祖先、两个分支尖端和合并结果"
    },
    {
        "filename": "git_flow.png",
        "prompt": "技术架构图：Git Flow工作流分支模型，展示main、develop、feature、release、hotfix五种分支类型及其关系，复杂的分支拓扑结构，专业配色"
    },
    
    # 第5篇
    {
        "filename": "team_workflow.png",
        "prompt": "技术流程图：团队协作标准流程，展示克隆项目、创建分支、开发提交、推送代码、创建PR、Code Review、合并分支的完整循环，步骤清晰，箭头连接"
    },
    {
        "filename": "create_pr.png",
        "prompt": "界面示意图：GitHub创建Pull Request界面，展示标题、描述、源分支、目标分支等关键表单元素的布局，模拟真实界面风格"
    },
    {
        "filename": "code_review.png",
        "prompt": "界面示意图：GitHub Code Review界面，展示代码差异对比、评论、批准按钮等关键元素，左右分栏显示代码变更，模拟真实界面"
    },
    
    # 第6篇
    {
        "filename": "fork_workflow.png",
        "prompt": "技术流程图：Fork工作流程，展示从原仓库Fork到个人账号、Clone到本地、创建分支、修改代码、Push到Fork、创建PR到原仓库的完整流程，循环结构"
    },
    {
        "filename": "create_pr_to_upstream.png",
        "prompt": "界面示意图：GitHub从Fork创建PR到原仓库界面，展示base repository和head repository的选择、PR标题描述表单，模拟真实界面风格"
    }
]

def main():
    # 设置输出目录
    output_dir = Path("d:/python_projects/NCT/images/Git入门专栏")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化客户端
    api_key = os.getenv('ZHIPU_API_KEY')
    if not api_key:
        print("错误：未设置 ZHIPU_API_KEY 环境变量")
        return
    
    client = ZhipuAiClient(api_key=api_key)
    
    print(f"准备生成 {len(IMAGES)} 张图片...")
    print(f"输出目录：{output_dir}")
    print("=" * 60)
    
    success_count = 0
    failed_count = 0
    
    for i, image_info in enumerate(IMAGES, 1):
        filename = image_info["filename"]
        prompt = image_info["prompt"]
        
        print(f"\n[{i}/{len(IMAGES)}] 生成：{filename}")
        print(f"  描述：{prompt[:80]}...")
        
        try:
            # 调用 GLM-Image API
            response = client.images.generations(
                model="glm-image",
                prompt=prompt,
                size="1280x1280"  # 正方形，适合技术图表
            )
            
            # 获取图片URL
            image_url = response.data[0].url
            print(f"  ✓ API 返回图片URL")
            
            # 下载图片
            img_response = requests.get(image_url)
            img_response.raise_for_status()
            
            # 保存图片
            save_path = output_dir / filename
            with open(save_path, 'wb') as f:
                f.write(img_response.content)
            
            print(f"  ✓ 已保存：{save_path}")
            success_count += 1
            
            # 等待一下避免请求过快
            import time
            time.sleep(2)
            
        except Exception as e:
            print(f"  ✗ 生成失败：{str(e)}")
            failed_count += 1
    
    # 打印总结
    print("\n" + "=" * 60)
    print("批量生成完成！")
    print(f"  成功：{success_count} 张")
    print(f"  失败：{failed_count} 张")
    print(f"  图片位置：{output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
