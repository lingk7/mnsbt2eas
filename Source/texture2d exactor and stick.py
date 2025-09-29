import cv2
import numpy as np
import os
import re
import sys
import glob
import time
from PIL import Image

def extract_sprites_from_texture2d(texture_path, sprite_dir, output_dir="output_part"):
    """从Texture2D图集中提取Sprite"""
    print("=" * 50)
    print("开始从Texture2D提取Sprite")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(texture_path) or not os.path.exists(sprite_dir):
        print("[ERROR] 文件或目录不存在")
        return False
    
    try:
        atlas = Image.open(texture_path)
        atlas_width, atlas_height = atlas.size
        print(f"图集尺寸: {atlas_width}x{atlas_height}")
        
        rect_pattern = re.compile(r"x: (\d+)\s+y: (\d+)\s+width: (\d+)\s+height: (\d+)")
        extracted_count = 0
        
        asset_files = [f for f in os.listdir(sprite_dir) if f.endswith(".asset")]
        print(f"找到 {len(asset_files)} 个Sprite定义文件")
        
        for file_name in asset_files:
            file_path = os.path.join(sprite_dir, file_name)

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        content = f.read()
                except Exception:
                    continue

            name_match = re.search(r"m_Name: ([^\n\r]+)", content)
            sprite_name = name_match.group(1).strip() if name_match else file_name.replace(".asset", "")

            output_path = os.path.join(output_dir, f"{sprite_name}.png")
            if os.path.exists(output_path):
                continue

            rect_match = rect_pattern.search(content)
            if not rect_match:
                continue

            x, y, width, height = map(int, rect_match.groups())
            adjusted_y = atlas_height - y - height
            box = (x, adjusted_y, x + width, adjusted_y + height)
            
            if (x + width > atlas_width or adjusted_y + height > atlas_height or 
                x < 0 or adjusted_y < 0 or width <= 0 or height <= 0):
                continue
                
            try:
                sprite = atlas.crop(box)
            except Exception:
                continue

            sprite.save(output_path)
            extracted_count += 1

        print(f"[OK] 成功提取 {extracted_count} 个Sprite")
        return extracted_count > 0
        
    except Exception as e:
        print(f"[ERROR] 提取Sprite时出错: {e}")
        return False

def load_image_with_alpha(path):
    """加载带透明通道的图像"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot load image: {path}")
    
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img

def transform_image(img, dx, dy, angle, scale, canvas_size):
    """变换图像（平移、旋转、缩放）"""
    (h, w) = img.shape[:2]
    image_center = (w / 2, h / 2)
    canvas_center = (canvas_size[0] / 2, canvas_size[1] / 2)

    M = cv2.getRotationMatrix2D(image_center, angle, scale)
    
    tx = canvas_center[0] - image_center[0] + dx
    ty = canvas_center[1] - image_center[1] + dy
    M[0, 2] += tx
    M[1, 2] += ty

    transformed = cv2.warpAffine(
        img, M, canvas_size,
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT
    )
    
    print(f"[DEBUG] 变换: dx={dx:.1f}, dy={dy:.1f}, 图像尺寸: {w}x{h}")
    
    return transformed

def get_layer_depth(image_name, image_data=None, apply_correction=True):
    """根据图像名称或SortingOrder确定图层深度"""
    # 如果有image_data且能找到对应的项，使用修正后的SortingOrder
    if image_data and apply_correction:
        for item in image_data:
            if item["name"] == image_name and "sorting_order" in item:
                sorting_order = item["sorting_order"]
                name_lower = image_name.lower()
                
                # 修正明显的顺序错误
                if 'eye' in name_lower:
                    # 眼睛应该在较高层级 (55-70)
                    if sorting_order < 50:  # 如果深度过低
                        return max(sorting_order, 60)
                    elif sorting_order > 70:  # 如果深度过高
                        return min(sorting_order, 65)
                
                elif 'mouth' in name_lower:
                    # 嘴巴应该在眼睛之下，头部基础之上 (45-60)
                    if sorting_order < 40:  # 如果深度过低（如23）
                        return max(sorting_order, 50)  # 提升到至少50
                    elif sorting_order > 60:  # 如果深度过高
                        return min(sorting_order, 55)
                
                elif 'headbase' in name_lower:
                    # 头部基础应该在面部特征之下 (30-45)
                    if sorting_order > 50:  # 如果深度过高（如52）
                        return min(sorting_order, 40)  # 降低到最多40
                
                elif any(part in name_lower for part in ['arml', 'armr']):
                    # 手臂应该在身体和头部之间 (20-35)
                    if sorting_order > 40:
                        return 25
                
                elif 'pale' in name_lower:
                    # 苍白特效应该在面部基础层之上 (40-50)
                    if sorting_order > 50:
                        return 45
                
                return sorting_order
    
    # 回退到基于名称的深度计算
    name_lower = image_name.lower()
    
    depth_map = {
        'shadow': 0,
        'body': 10,
        'arml': 20,
        'armr': 20,
        'arms': 20,
        'headbase': 30,
        'pale': 40,      # 面部特效在头部基础之上
        'cheek': 50,     # 脸颊
        'mouth': 55,     # 嘴巴 - 调整到比眼睛低但比头部基础高
        'eye': 60,       # 眼睛
        'hair': 80,
        'clippingmask': 90,
    }
    
    for keyword, depth in depth_map.items():
        if keyword in name_lower:
            return depth
    
    return 25

def composite_images(image_data, included_names, canvas_size, image_folder, image_ext=".png"):
    """合成多个图像，按照正确的图层顺序"""
    canvas = np.zeros((canvas_size[1], canvas_size[0], 4), dtype=np.uint8)
    name_to_data = {item["name"]: item for item in image_data}
    
    # 检查是否为模式三的手动深度设置
    if included_names and isinstance(included_names[0], dict) and 'manual_depth' in included_names[0]:
        # 模式三：使用手动设置的深度顺序
        sorted_items = sorted(included_names, key=lambda x: x['manual_depth'])
        sorted_names = [item['name'] for item in sorted_items]
        print("[模式三] 使用手动设置的深度顺序")
        
        # 显示最终合成顺序
        print("\n最终合成顺序 (从底到顶):")
        for item in sorted_items:
            print(f"  {item['name']}: 深度 {item['manual_depth']}")
    else:
        # 其他模式：使用自动深度计算
        # 检查是否是模式一（需要应用修正）
        apply_correction = not any(isinstance(name, dict) and name.get('mode') == 3 for name in included_names)
        sorted_names = sorted(included_names, key=lambda name: get_layer_depth(name, image_data, apply_correction))
    
    print("合成部件详情:")
    print("-" * 50)
    
    for name in sorted_names:
        # 对于模式三，我们已经知道深度，不需要重新计算
        if included_names and isinstance(included_names[0], dict) and 'manual_depth' in included_names[0]:
            # 找到对应的手动深度设置
            manual_item = next((item for item in included_names if item['name'] == name), None)
            depth = manual_item['manual_depth'] if manual_item else get_layer_depth(name, image_data, False)
            source = "手动设置"
        else:
            depth = get_layer_depth(name, image_data, apply_correction)
            source = "SortingOrder" if any(item.get("sorting_order") for item in image_data if item["name"] == name) else "名称推断"
        
        if name not in name_to_data:
            print(f"[SKIP] {name}: 未找到坐标数据")
            continue

        item = name_to_data[name]
        image_path = os.path.join(image_folder, name + image_ext)
        
        if not os.path.exists(image_path):
            print(f"[MISS] {name}: 图像文件不存在")
            continue
            
        try:
            img = load_image_with_alpha(image_path)
            print(f"[LOAD] {name}: 尺寸 {img.shape[1]}x{img.shape[0]}, 深度 {depth} ({source})")
        except FileNotFoundError:
            print(f"[ERROR] {name}: 加载失败")
            continue

        transformed = transform_image(img, item["dx"], item["dy"], item["angle"], item["scale"], canvas_size)

        if transformed.shape[2] == 4:
            alpha = transformed[:, :, 3:4] / 255.0
            canvas[:, :, :3] = (1 - alpha) * canvas[:, :, :3] + alpha * transformed[:, :, :3]
            canvas[:, :, 3:] = np.maximum(canvas[:, :, 3:], transformed[:, :, 3:])
        else:
            canvas[:, :, :3] = transformed
            canvas[:, :, 3] = 255
            
        print(f"[APPLY] {name}: dx={item['dx']:.1f}, dy={item['dy']:.1f}")

    print("-" * 50)
    return canvas

def composite_multiple_sets(image_data, included_names_list, canvas_size, image_folder, 
                           outputname="composite", output_folder="output", image_ext=".png"):
    """合成多组图像"""
    os.makedirs(output_folder, exist_ok=True)
    
    if not os.path.exists(image_folder):
        print("[ERROR] 图像文件夹不存在")
        return

    timestamp = int(time.time())
    
    for i, included_names in enumerate(included_names_list):
        print(f"\n合成第 {i+1} 组图像...")
        
        # 获取组合描述
        if isinstance(included_names, dict):
            description = included_names.get('description', f'组合_{i+1}')
            components = included_names.get('components', [])
            has_arms = included_names.get('has_arms', False)
        else:
            description = f'Combination_{i+1}'
            components = included_names
            has_arms = any(any(arm_keyword in comp for arm_keyword in ['ArmL', 'ArmR', 'Arms']) for comp in components)
        
        print(f"组合描述: {description}")
        
        # 只为模式一保留自动补充身体部件，并且只在没有手臂时才补充手臂
        if isinstance(included_names, dict) and included_names.get('mode') == 1:
            # 如果已经标记包含手臂，确保自动补充逻辑知道这一点
            temp_components = components.copy()
            if has_arms:
                # 临时添加一个标记，让auto_supplement_body_parts知道有手臂
                temp_components.append("_HAS_ARMS_MARKER_")
            components = auto_supplement_body_parts(temp_components, image_folder)
            # 移除标记
            components = [comp for comp in components if comp != "_HAS_ARMS_MARKER_"]
        
        result = composite_images(image_data, components, canvas_size, image_folder, image_ext)

        # 使用描述作为输出文件名的一部分
        safe_description = re.sub(r'[^\w\-_]+', '_', description)
        output_name = f"{outputname}_{safe_description}_{timestamp}.png"
        output_path = os.path.join(output_folder, output_name)
        cv2.imwrite(output_path, result)
        print(f"[OK] 已保存: {output_path}")
        
def parse_prefab_yaml(file_path):
    """解析Unity Prefab YAML文件，提取GameObject名称、位置坐标和SortingOrder"""
    results = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                content = file.read()
        except Exception as e:
            return results
    
    documents = content.split('---')
    game_objects = {}
    transforms = {}
    sprite_renderers = {}
    
    for doc in documents:
        if not doc.strip():
            continue
            
        file_id_match = re.search(r'&(\d+)', doc)
        if not file_id_match:
            continue
            
        file_id = file_id_match.group(1)
        
        if 'GameObject:' in doc and 'm_Name:' in doc:
            name_match = re.search(r'm_Name:\s*([^\r\n]+)', doc)
            if name_match:
                game_objects[file_id] = name_match.group(1).strip()
        
        elif 'Transform:' in doc and 'm_LocalPosition:' in doc:
            pos_match = re.search(r'm_LocalPosition:\s*\{([^}]+)\}', doc)
            if pos_match:
                pos_str = pos_match.group(1)
                x_match = re.search(r'x:\s*([\d.-]+)', pos_str)
                y_match = re.search(r'y:\s*([\d.-]+)', pos_str)
                z_match = re.search(r'z:\s*([\d.-]+)', pos_str)
                
                if x_match and y_match and z_match:
                    position = {
                        'x': float(x_match.group(1)),
                        'y': float(y_match.group(1)),
                        'z': float(z_match.group(1))
                    }
                    transforms[file_id] = position
        
        elif 'SpriteRenderer:' in doc:
            sorting_order_match = re.search(r'm_SortingOrder:\s*(\d+)', doc)
            if sorting_order_match:
                sorting_order = int(sorting_order_match.group(1))
                sprite_renderers[file_id] = sorting_order
    
    for doc in documents:
        if not doc.strip():
            continue
            
        file_id_match = re.search(r'&(\d+)', doc)
        if not file_id_match:
            continue
            
        file_id = file_id_match.group(1)
        
        if file_id in game_objects:
            comp_ids = re.findall(r'component:\s*\{fileID:\s*(\d+)\}', doc)
            
            transform_id = None
            sprite_renderer_id = None
            
            for comp_id in comp_ids:
                if comp_id in transforms:
                    transform_id = comp_id
                if comp_id in sprite_renderers:
                    sprite_renderer_id = comp_id
            
            if transform_id:
                obj_name = game_objects[file_id]
                position = transforms[transform_id]
                sorting_order = sprite_renderers.get(sprite_renderer_id, 0)
                
                results.append({
                    'name': obj_name,
                    'position': position,
                    'sorting_order': sorting_order,
                    'file_id': file_id,
                    'transform_id': transform_id
                })
    
    return results

def auto_supplement_body_parts(components, image_folder):
    """自动补充缺失的基础身体部件，智能判断手臂部件补充"""
    available_images = get_available_images(image_folder)
    
    # 检查是否已包含手臂部件（包括Arms组合）
    has_arms = any(any(arm_keyword in comp for arm_keyword in ['ArmL', 'ArmR', 'Arms']) for comp in components)
    
    # 检测当前组合使用的头部版本
    head_version = detect_head_version(components, available_images)
    
    # 基础部件列表
    base_parts = [
        'Body',           # 基础身体
        f'HeadBase{head_version}',  # 对应版本的头部基础
    ]
    
    # 只有在完全没有手臂部件时才补充基础手臂
    if not has_arms:
        # 尝试补充最基础的手臂组合
        base_arms = find_base_arm_combination(available_images)
        if base_arms:
            base_parts.extend(base_arms)
            print(f"[INFO] 自动补充手臂部件: {base_arms}")
    
    supplementary_parts = []
    
    for base_part in base_parts:
        part_exists = any(base_part == comp for comp in components)
        if not part_exists:
            # 精确匹配基础部件
            if base_part in available_images:
                supplementary_parts.append(base_part)
            else:
                # 如果精确匹配失败，尝试寻找最接近的基础版本
                close_matches = find_closest_base_part(base_part, available_images)
                if close_matches:
                    supplementary_parts.append(close_matches[0])
    
    if supplementary_parts:
        print(f"[INFO] 自动补充身体部件: {supplementary_parts}")
        return components + supplementary_parts
    
    return components

def find_base_arm_combination(available_images):
    """寻找最基础的手臂组合"""
    # 优先寻找双臂组合
    arms_versions = [img for img in available_images if img.startswith('Arms') and re.match(r'Arms\d+$', img)]
    if arms_versions:
        return [sorted(arms_versions)[0]]  # 返回编号最小的双臂
    
    # 其次寻找对应的左右手臂
    arml_versions = sorted([img for img in available_images if img.startswith('ArmL') and re.match(r'ArmL\d+$', img)])
    armr_versions = sorted([img for img in available_images if img.startswith('ArmR') and re.match(r'ArmR\d+$', img)])
    
    if arml_versions and armr_versions:
        # 尝试找到对应的版本
        for arml in arml_versions:
            arml_num = arml.replace('ArmL', '')
            corresponding_armr = f"ArmR{arml_num}"
            if corresponding_armr in armr_versions:
                return [arml, corresponding_armr]
        
        # 如果没有完全对应的版本，使用各自的最小版本
        return [arml_versions[0], armr_versions[0]]
    
    # 如果只有单边手臂，补充可用的
    if arml_versions:
        return [arml_versions[0]]
    if armr_versions:
        return [armr_versions[0]]
    
    return []
    
def find_closest_base_part(base_part, available_images):
    """寻找最接近的基础部件版本"""
    if 'HeadBase' in base_part:
        # 对于头部基础，寻找其他版本
        headbase_versions = [img for img in available_images if img.startswith('HeadBase')]
        return sorted(headbase_versions)  # 返回排序后的版本
    
    elif base_part == 'Body':
        # 对于身体，寻找基础身体版本
        body_versions = [img for img in available_images if img in ['Body', 'Body01']]
        return body_versions
    
    elif base_part == 'ArmL01':
        # 对于左臂，寻找基础左臂版本
        arml_versions = [img for img in available_images if img.startswith('ArmL') and len(img) <= 6]  # ArmL01, ArmL02等
        return sorted(arml_versions)
    
    elif base_part == 'ArmR01':
        # 对于右臂，寻找基础右臂版本
        armr_versions = [img for img in available_images if img.startswith('ArmR') and len(img) <= 6]  # ArmR01, ArmR02等
        return sorted(armr_versions)
    
    return []

def detect_head_version(components, available_images):
    """检测当前组合使用的头部版本"""
    # 从现有组件中检测头部版本
    for comp in components:
        if any(head_keyword in comp for head_keyword in ['Head', 'OptionB_Head', 'OptionF_Head']):
            # 提取版本号，如 OptionB_Head02 -> 02
            version_match = re.search(r'Head(\d+)', comp)
            if version_match:
                return version_match.group(1)
    
    # 从可用图像中推断最常见的头部版本
    head_versions = {}
    for img in available_images:
        if 'HeadBase' in img:
            version_match = re.search(r'HeadBase(\d+)', img)
            if version_match:
                version = version_match.group(1)
                head_versions[version] = head_versions.get(version, 0) + 1
    
    if head_versions:
        # 返回出现次数最多的版本
        return max(head_versions.items(), key=lambda x: x[1])[0]
    
    return "01"  # 默认版本

def select_best_match(part_keyword, matching_parts, existing_components):
    """选择最匹配的部件"""
    if len(matching_parts) == 1:
        return matching_parts[0]
    
    # 如果有多个匹配，优先选择与现有组件版本一致的
    detected_version = detect_head_version(existing_components, matching_parts)
    versioned_parts = [part for part in matching_parts if f"{part_keyword}{detected_version}" in part]
    
    if versioned_parts:
        return versioned_parts[0]
    
    # 否则返回第一个匹配项
    return matching_parts[0]

def update_image_data_from_prefab(image_data, prefab_file, ratio=100):
    """从Unity Prefab文件更新image_data中的坐标信息和SortingOrder"""
    if not os.path.exists(prefab_file):
        print("[ERROR] Prefab文件不存在")
        return image_data
    
    print(f"解析Prefab文件: {prefab_file}")
    
    try:
        prefab_data = parse_prefab_yaml(prefab_file)
        
        if not prefab_data:
            print("未找到有效的坐标信息")
            return image_data
        
        prefab_dict = {item['name']: item for item in prefab_data}
        updated_count = 0
        
        for item in image_data:
            name = item["name"]
            if name in prefab_dict:
                prefab_item = prefab_dict[name]
                position = prefab_item['position']
                item["dx"] = position['x'] * ratio
                item["dy"] = position['y'] * -ratio
                updated_count += 1
                
                if 'sorting_order' in prefab_item:
                    item["sorting_order"] = prefab_item['sorting_order']
        
        print(f"[OK] 更新了 {updated_count} 个图像项的坐标")
        
        # 显示坐标调试信息
        print("\n坐标调试信息:")
        print("-" * 40)
        for item in image_data[:5]:
            if 'dx' in item and 'dy' in item:
                print(f"{item['name']}: dx={item['dx']:.2f}, dy={item['dy']:.2f}")
        if len(image_data) > 5:
            print(f"... 还有 {len(image_data) - 5} 项")
        print("-" * 40)
        
        return image_data
        
    except Exception as e:
        print(f"[ERROR] 处理Prefab文件时出错: {e}")
        return image_data

def get_available_images(image_folder):
    """获取可用的图像文件列表（不含扩展名）"""
    if not os.path.exists(image_folder):
        return []
    
    png_files = [os.path.splitext(f)[0] for f in os.listdir(image_folder) 
                if f.endswith('.png') and os.path.isfile(os.path.join(image_folder, f))]
    
    return png_files

def auto_add_missing_images(image_data, image_folder, ratio=100):
    """自动检测并添加image_data中缺失的图像项"""
    if not os.path.exists(image_folder):
        return image_data
    
    png_files = get_available_images(image_folder)
    existing_names = set(item["name"] for item in image_data)
    missing_names = set(png_files) - existing_names
    
    if not missing_names:
        return image_data
    
    print(f"[INFO] 添加 {len(missing_names)} 个新图像到image_data")
    
    new_items = []
    for name in sorted(missing_names):
        default_dx, default_dy = guess_default_position(name, ratio)
        
        new_item = {
            "name": name,
            "dx": default_dx,
            "dy": default_dy,
            "angle": 0,
            "scale": 1.0
        }
        new_items.append(new_item)
    
    image_data.extend(new_items)
    return image_data

def guess_default_position(name, ratio):
    """根据图像名称猜测默认位置"""
    default_x, default_y = 0, 0
    name_lower = name.lower()
    
    if any(part in name_lower for part in ['body', 'torso']):
        default_y = -3.5 * ratio
    elif any(part in name_lower for part in ['head', 'face', 'cheek', 'eye', 'mouth']):
        default_y = 14 * ratio
    elif 'arml' in name_lower:
        default_x = 4 * ratio
        default_y = 5 * ratio
    elif 'armr' in name_lower:
        default_x = -4 * ratio
        default_y = 5 * ratio
    elif 'shadow' in name_lower:
        default_y = 6 * ratio
    elif any(effect in name_lower for effect in ['sweat', 'pale']):
        default_y = 13 * ratio
    else:
        default_y = 7 * ratio
    
    return default_x, -default_y

def is_complete_face_combination(components):
    """判断组合是否包含完整的五官（眼睛和嘴巴）"""
    has_eye = any('eye' in comp.lower() for comp in components)
    has_mouth = any('mouth' in comp.lower() for comp in components)
    return has_eye and has_mouth

def parse_prefab_compositions(file_path, available_images):
    """修正后的Prefab Composition解析，确保正确解析头部版本"""
    compositions = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                content = file.read()
        except Exception:
            return compositions
    
    # 提取所有Key-Composition对
    raw_compositions = {}
    key_pattern = re.compile(r'- Key:\s*([^\r\n]+)')
    comp_pattern = re.compile(r'Composition:\s*([^\r\n]+)')
    
    lines = content.split('\n')
    current_key = None
    
    for line in lines:
        line = line.strip()
        
        # 查找Key
        key_match = key_pattern.search(line)
        if key_match:
            current_key = key_match.group(1).strip()
        
        # 查找Composition
        comp_match = comp_pattern.search(line)
        if comp_match and current_key:
            comp_str = comp_match.group(1).strip()
            # 按逗号分割，但保留路径结构
            comp_items = [item.strip() for item in comp_str.split(',')]
            raw_compositions[current_key] = comp_items
            current_key = None
    
    # 递归解析组合，确保头部版本一致性
    def resolve_composition(key, visited=None):
        if visited is None:
            visited = set()
        
        if key in visited:
            return []
        visited.add(key)
        
        result = []
        
        # 如果key不在原始组合中，检查是否是最终部件
        if key not in raw_compositions:
            # 处理关闭项
            if key.endswith('-'):
                return []
            
            # 处理替换关系 A>B
            if '>' in key:
                base, replacement = key.split('>')
                # 取替换后的名称
                final_name = replacement.strip()
                if final_name in available_images:
                    result.append(final_name)
                return result
            
            # 检查是否是最终部件名称
            if key in available_images:
                result.append(key)
            return result
        
        # 检测当前组合的头部版本
        head_version = None
        for item in raw_compositions[key]:
            if 'Head' in item and 'Base' not in item:
                version_match = re.search(r'Head(\d+)', item)
                if version_match:
                    head_version = version_match.group(1)
                    break
        
        # 解析组合中的每个项目
        for item in raw_compositions[key]:
            # 跳过关闭项
            if item.endswith('-'):
                continue
            
            # 处理替换关系
            if '>' in item:
                base, replacement = item.split('>')
                base = base.strip()
                replacement = replacement.strip()
                
                # 如果替换名称在可用图像中，使用它
                if replacement in available_images:
                    result.append(replacement)
                # 否则检查基础名称
                elif base in available_images:
                    result.append(base)
                # 如果都不是，递归解析
                elif base in raw_compositions:
                    result.extend(resolve_composition(base, visited))
                elif replacement in raw_compositions:
                    result.extend(resolve_composition(replacement, visited))
            else:
                # 普通项目，可能是最终部件或另一个组合
                if item in available_images:
                    # 确保头部基础版本与头部版本一致
                    if 'HeadBase' in item and head_version:
                        correct_headbase = f"HeadBase{head_version}"
                        if correct_headbase in available_images and correct_headbase != item:
                            result.append(correct_headbase)
                        else:
                            result.append(item)
                    else:
                        result.append(item)
                elif item in raw_compositions:
                    result.extend(resolve_composition(item, visited))
        
        visited.remove(key)
        return list(set(result))  # 去重
    
    # 解析所有组合，只保留完整表情组合
    for key in raw_compositions:
        resolved = resolve_composition(key)
        if resolved and is_complete_face_combination(resolved):
            compositions[key] = resolved
    
    return compositions

def interactive_composition_selection(compositions_dict, all_available_images, image_data):
    """交互式选择表情组合"""
    if not compositions_dict:
        print("[ERROR] 没有可用的表情组合")
        return []
    
    print("\n" + "=" * 50)
    print("选择合成模式")
    print("=" * 50)
    print("1. 选择预设组合 (自动补全基础身体部件)")
    print("2. 自定义组合 (手动选择部件)")
    print("3. 手动深度编辑 (手动选择部件和深度)")
    
    mode = input("请选择模式 (1/2/3, 默认1): ").strip() or "1"
    
    if mode == "1":
        return preset_composition_selection(compositions_dict, all_available_images)
    elif mode == "2":
        return custom_composition_selection(all_available_images)
    elif mode == "3":
        return manual_depth_composition_selection(all_available_images, image_data)
    else:
        print("[ERROR] 无效的模式选择")
        return []

def preset_composition_selection(compositions_dict, all_available_images):
    """模式一：使用预设组合（自动补全基础身体部件）"""
    print("\n" + "=" * 50)
    print("预设组合模式")
    print("=" * 50)
    print("此模式会自动选择完整的表情组合并补全基础身体部件")
    print("包含基础手臂组合选项，避免自动补充错误的手臂部件")
    print("注意：多个组合将合并为一张图片")
    
    # 只显示完整表情组合
    complete_compositions = {}
    for key, components in compositions_dict.items():
        if is_complete_face_combination(components):
            complete_compositions[key] = components
    
    if not complete_compositions:
        print("没有找到完整表情组合")
        return []
    
    # 对组合键进行排序
    sorted_keys = sorted(complete_compositions.keys())
    index_map = {}
    
    print("\n可用的预设表情组合:")
    print("-" * 40)
    
    for i, key in enumerate(sorted_keys, 1):
        index_map[i] = key
        components = complete_compositions[key]
        
        # 提取主要特征用于显示
        features = []
        for comp in components:
            if 'Eye' in comp:
                eye_feature = comp.replace('Eyes', '').replace('01', '').replace('02', '').replace('_', ' ')
                features.append(f"眼:{eye_feature}")
            elif 'Mouth' in comp:
                mouth_feature = comp.replace('Mouth', '').replace('01', '').replace('02', '').replace('_', ' ')
                features.append(f"嘴:{mouth_feature}")
        
        feature_desc = " + ".join(features[:2])  # 显示主要特征
        print(f"{i:2d}. {key:15} | {feature_desc}")
    
    # 添加基础手臂组合选项
    arm_combinations = generate_arm_combinations(all_available_images)
    
    print("\n基础手臂组合选项:")
    print("-" * 40)
    
    arm_start_index = len(sorted_keys) + 1
    for i, (combo_name, arm_components) in enumerate(arm_combinations.items(), arm_start_index):
        index_map[i] = combo_name
        print(f"{i:2d}. {combo_name:15} | 手臂组合: {', '.join(arm_components)}")
    
    selected_combinations = []
    
    print(f"\n请选择组合编号 (1-{len(index_map)})，多个编号用逗号分隔:")
    print("注意：多个组合将合并为一张图片")
    try:
        selected_indices = input("组合编号: ").strip()
        if selected_indices:
            indices = [int(x.strip()) for x in selected_indices.split(',')]
            
            # 合并所有选中的组合为一个组合
            all_components = []
            combo_descriptions = []
            
            for i in indices:
                if i in index_map:
                    combo_name = index_map[i]
                    
                    if combo_name in arm_combinations:
                        # 选择的是手臂组合
                        arm_combo = arm_combinations[combo_name]
                        all_components.extend(arm_combo)
                        combo_descriptions.append(combo_name)
                        print(f"已选择: {combo_name}")
                    else:
                        # 选择的是表情组合
                        all_components.extend(compositions_dict[combo_name])
                        combo_descriptions.append(combo_name)
                        print(f"已选择: {combo_name}")
            
            # 去重
            all_components = list(set(all_components))
            
            if all_components:
                # 创建一个合并的组合
                merged_description = " + ".join(combo_descriptions)
                selected_combinations.append({
                    'mode': 1,
                    'components': all_components,
                    'description': merged_description
                })
                print(f"合并为: {merged_description}")
        elif index_map:
            # 默认选择第一个表情组合
            selected_combinations.append({
                'mode': 1,
                'components': compositions_dict[index_map[1]],
                'description': index_map[1]
            })
            print(f"默认选择: {index_map[1]}")
    except ValueError:
        if index_map:
            selected_combinations.append({
                'mode': 1,
                'components': compositions_dict[index_map[1]],
                'description': index_map[1]
            })
            print(f"默认选择: {index_map[1]}")
    
    return selected_combinations

def generate_arm_combinations(available_images):
    """生成稳定的手臂组合"""
    arm_combinations = {}
    
    # 查找可用的手臂部件
    arml_versions = sorted([img for img in available_images if img.startswith('ArmL') and re.match(r'ArmL\d+$', img)])
    armr_versions = sorted([img for img in available_images if img.startswith('ArmR') and re.match(r'ArmR\d+$', img)])
    arms_versions = sorted([img for img in available_images if img.startswith('Arms') and re.match(r'Arms\d+$', img)])
    
    # 生成左右手臂对应组合
    for arml in arml_versions:
        arml_num = arml.replace('ArmL', '')
        corresponding_armr = f"ArmR{arml_num}"
        if corresponding_armr in armr_versions:
            combo_name = f"Arms_{arml_num}"
            arm_combinations[combo_name] = [arml, corresponding_armr]
    
    # 添加双臂组合
    for arms in arms_versions:
        arms_num = arms.replace('Arms', '')
        combo_name = f"ArmsCombo_{arms_num}"
        arm_combinations[combo_name] = [arms]
    
    # 如果没有找到对应组合，创建基础组合
    if not arm_combinations:
        if arml_versions and armr_versions:
            arm_combinations["Arms_01"] = [arml_versions[0], armr_versions[0]]
        elif arms_versions:
            arm_combinations["ArmsCombo_01"] = [arms_versions[0]]
    
    return arm_combinations

def find_default_face_combo(compositions_dict):
    """寻找默认的表情组合"""
    # 优先寻找Normal相关的组合
    for key, components in compositions_dict.items():
        if 'Normal' in key and is_complete_face_combination(components):
            return components
    
    # 如果没有Normal，返回第一个完整表情组合
    for key, components in compositions_dict.items():
        if is_complete_face_combination(components):
            return components
    
    return None

def custom_composition_selection(all_available_images):
    """模式二：自定义组合（包含Arms部件选项）"""
    print("\n" + "=" * 50)
    print("自定义组合 (模式二)")
    print("=" * 50)
    
    # 对部件进行分类显示
    categories = {
        '身体基础': ['Body', 'Body01'],
        '头部基础': [img for img in all_available_images if 'HeadBase' in img],
        '左手臂': [img for img in all_available_images if img.startswith('ArmL') and re.match(r'ArmL\d+$', img)],
        '右手臂': [img for img in all_available_images if img.startswith('ArmR') and re.match(r'ArmR\d+$', img)],
        '双臂组合': [img for img in all_available_images if img.startswith('Arms') and re.match(r'Arms\d+$', img)],
        '眼睛': [img for img in all_available_images if 'Eye' in img],
        '嘴巴': [img for img in all_available_images if 'Mouth' in img],
        '脸颊': [img for img in all_available_images if 'Cheek' in img],
        '其他': [img for img in all_available_images if not any(keyword in img for keyword in 
                ['Body', 'HeadBase', 'ArmL', 'ArmR', 'Arms', 'Eye', 'Mouth', 'Cheek'])]
    }
    
    component_map = {}
    current_index = 1
    
    print("\n所有可用部件（按类别分类）:")
    print("-" * 50)
    
    for category, items in categories.items():
        if items:
            print(f"\n{category}:")
            print("-" * 20)
            for item in sorted(items):
                component_map[current_index] = item
                print(f"{current_index:3d}. {item}")
                current_index += 1
    
    selected_combinations = []
    
    print(f"\n请输入要组合的部件编号（用逗号分隔）:")
    try:
        selected_indices = input("部件编号: ").strip()
        if selected_indices:
            indices = [int(x.strip()) for x in selected_indices.split(',')]
            custom_combo = [component_map[i] for i in indices if i in component_map]
            if custom_combo:
                selected_combinations.append(custom_combo)
    except ValueError:
        print("[ERROR] 输入格式错误")
    
    return selected_combinations

def manual_depth_composition_selection(all_available_images, image_data):
    """模式三：手动深度编辑（不使用修正）"""
    print("\n" + "=" * 50)
    print("手动深度编辑 (模式三)")
    print("=" * 50)
    
    print("\n所有可用部件")
    print("-" * 20)
    all_components = sorted(all_available_images)
    component_map = {}
    for i, comp in enumerate(all_components, 1):
        # 模式三不使用修正
        current_depth = get_layer_depth(comp, image_data, apply_correction=False)
        depth_source = "SortingOrder" if any(item.get("sorting_order") for item in image_data if item["name"] == comp) else "名称推断"
        print(f"{i:3d}. {comp} [深度: {current_depth} - {depth_source}]")
        component_map[i] = comp
    
    selected_combinations = []
    
    print("\n请输入要组合的部件编号（用逗号分隔）:")
    try:
        selected_indices = input("部件编号: ").strip()
        if selected_indices:
            indices = [int(x.strip()) for x in selected_indices.split(',')]
            selected_components = [component_map[i] for i in indices if i in component_map]
            
            if selected_components:
                manual_combo = []
                print("\n手动设置部件深度 (输入数字，空值使用默认深度):")
                for comp in selected_components:
                    # 模式三不使用修正
                    current_depth = get_layer_depth(comp, image_data, apply_correction=False)
                    depth_input = input(f"  {comp} 的深度 (当前: {current_depth}): ").strip()
                    
                    if depth_input:
                        try:
                            manual_depth = int(depth_input)
                            manual_combo.append({
                                'name': comp,
                                'manual_depth': manual_depth
                            })
                            print(f"    -> 设置深度为: {manual_depth}")
                        except ValueError:
                            manual_combo.append({
                                'name': comp,
                                'manual_depth': current_depth
                            })
                            print(f"    -> 使用默认深度: {current_depth}")
                    else:
                        manual_combo.append({
                            'name': comp,
                            'manual_depth': current_depth
                        })
                        print(f"    -> 使用默认深度: {current_depth}")
                
                manual_combo_sorted = sorted(manual_combo, key=lambda x: x['manual_depth'])
                final_combo = [item['name'] for item in manual_combo_sorted]
                
                print("\n最终部件顺序:")
                for item in manual_combo_sorted:
                    print(f"  {item['name']}: 深度 {item['manual_depth']}")
                
                selected_combinations.append(manual_combo_sorted)
    except ValueError:
        print("[ERROR] 输入格式错误")
    
    return selected_combinations

def find_texture2d_and_sprite_dirs():
    """自动查找Texture2D和Sprite目录"""
    texture_dirs = ["Assets/Texture2D", "Texture2D", "Assets/Textures", "Textures", "."]
    sprite_dirs = ["Assets/Sprite", "Sprite", "Assets/Sprites", "Sprites", "."]
    
    texture_files = []
    sprite_dirs_found = []
    
    for texture_dir in texture_dirs:
        if os.path.exists(texture_dir):
            png_files = glob.glob(os.path.join(texture_dir, "*.png"))
            texture_files.extend(png_files)
    
    for sprite_dir in sprite_dirs:
        if os.path.exists(sprite_dir):
            asset_files = glob.glob(os.path.join(sprite_dir, "*.asset"))
            if asset_files:
                sprite_dirs_found.append(sprite_dir)
    
    return texture_files, sprite_dirs_found

def find_prefab_files():
    """自动扫描Prefab文件"""
    unity_paths = [
        os.path.join("ExportedProject", "Assets", "#WitchTrials", "Prefabs", "Naninovel", "Characters", "LayeredCharacters"),
        os.path.join("Assets", "#WitchTrials", "Prefabs", "Naninovel", "Characters", "LayeredCharacters"),
        os.path.join("#WitchTrials", "Prefabs", "Naninovel", "Characters", "LayeredCharacters"),
        "."
    ]
    
    prefab_files = []
    for path in unity_paths:
        if os.path.exists(path):
            found_files = glob.glob(os.path.join(path, "*.prefab"))
            prefab_files.extend(found_files)
    
    if not prefab_files:
        prefab_files = glob.glob("*.prefab")
    
    return prefab_files

def main():
    ratio = 100
    image_data = [
        {"name": "ArmL01", "dx": 5.57 * ratio, "dy": 3.025 * -ratio, "angle": 0, "scale": 1.0},
    ]

    canvas_size = (2000, 4000)
    
    print("=" * 50)
    print("分层角色图像合成工具")
    print("=" * 50)
    
    image_folder = "output_part"
    
    if os.path.exists(image_folder) and get_available_images(image_folder):
        print("[OK] 使用现有图像文件夹")
    else:
        texture_files, sprite_dirs = find_texture2d_and_sprite_dirs()
        
        if texture_files and sprite_dirs:
            print("\n发现Texture2D和Sprite资源")
            
            extract_choice = input("是否从Texture2D提取Sprite？(y/n, 默认y): ").strip().lower()
            if extract_choice in ['', 'y', 'yes', '是']:
                if len(texture_files) == 1:
                    texture_path = texture_files[0]
                else:
                    print("\n选择Texture2D文件:")
                    for i, file in enumerate(texture_files, 1):
                        print(f"{i}. {file}")
                    try:
                        choice = input("文件编号 (默认1): ").strip()
                        idx = int(choice) - 1 if choice else 0
                        texture_path = texture_files[idx] if 0 <= idx < len(texture_files) else texture_files[0]
                    except ValueError:
                        texture_path = texture_files[0]
                
                if len(sprite_dirs) == 1:
                    sprite_dir = sprite_dirs[0]
                else:
                    print("\n选择Sprite目录:")
                    for i, dir_path in enumerate(sprite_dirs, 1):
                        print(f"{i}. {dir_path}")
                    try:
                        choice = input("目录编号 (默认1): ").strip()
                        idx = int(choice) - 1 if choice else 0
                        sprite_dir = sprite_dirs[idx] if 0 <= idx < len(sprite_dirs) else sprite_dirs[0]
                    except ValueError:
                        sprite_dir = sprite_dirs[0]
                
                extract_sprites_from_texture2d(texture_path, sprite_dir, image_folder)
        else:
            print("[ERROR] 未找到Texture2D和Sprite资源")
            return
    
    available_images = get_available_images(image_folder)
    if not available_images:
        print("[ERROR] 图像文件夹中没有PNG文件")
        return
    
    print(f"[OK] 找到 {len(available_images)} 个图像文件")
    
    image_data = auto_add_missing_images(image_data, image_folder, ratio)
    
    prefab_files = find_prefab_files()
    prefab_file = None
    
    if prefab_files:
        if len(prefab_files) == 1:
            prefab_file = prefab_files[0]
            print(f"使用prefab文件: {prefab_file}")
        else:
            print("\n选择prefab文件:")
            for i, file in enumerate(prefab_files, 1):
                print(f"{i}. {file}")
            try:
                choice = input("文件编号 (默认1): ").strip()
                idx = int(choice) - 1 if choice else 0
                if 0 <= idx < len(prefab_files):
                    prefab_file = prefab_files[idx]
                else:
                    prefab_file = prefab_files[0]
            except ValueError:
                prefab_file = prefab_files[0]
    
    if prefab_file:
        image_data = update_image_data_from_prefab(image_data, prefab_file, ratio)
    
    compositions_dict = {}
    if prefab_file and os.path.exists(prefab_file):
        compositions_dict = parse_prefab_compositions(prefab_file, available_images)
    
    if not compositions_dict:
        print("使用默认表情组合")
        default_compositions = {}
        
        body = next((img for img in available_images if "Body" in img), None)
        armL = next((img for img in available_images if "ArmL" in img), None)
        armR = next((img for img in available_images if "ArmR" in img), None)
        head = next((img for img in available_images if "Head" in img and "Base" in img), None)
        
        if body and armL and armR and head:
            normal_combo = [body, armL, armR, head]
            facial = next((img for img in available_images if "FacialLineDrawing" in img), None)
            if facial: normal_combo.append(facial)
            
            default_compositions["Default"] = normal_combo
        
        compositions_dict = default_compositions
    
    while True:
        selected_combinations = interactive_composition_selection(compositions_dict, available_images, image_data)
        
        if not selected_combinations:
            print("[ERROR] 没有选择任何组合")
        else:
            print(f"\n合成 {len(selected_combinations)} 个组合")
            composite_multiple_sets(image_data, selected_combinations, canvas_size, image_folder, "composite")
            print("[OK] 合成完成")
        
        continue_choice = input("\n是否继续合成其他组合？(y/n, 默认n): ").strip().lower()
        if continue_choice not in ['y', 'yes', '是']:
            break
    
    print("\n程序执行完成")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("使用方法:")
        print("  python stick.py                    # 使用默认坐标数据")
        print("  python stick.py <prefab_file>      # 从Prefab文件更新坐标数据")
        sys.exit(0)
    
    main()
