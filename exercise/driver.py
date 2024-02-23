import torch

# Получение количества доступных устройств
device_count = torch.cuda.device_count()

# Печать информации о каждом устройстве
for device_id in range(device_count):
    device_name = torch.cuda.get_device_name(device_id)
    print(f"Устройство {device_id}: {device_name}")

# Печать информации о текущем устройстве (если GPU доступно)
if torch.cuda.is_available():
    current_device = torch.cuda.current_device()
    current_device_name = torch.cuda.get_device_name(current_device)
    print(f"\nТекущее устройство: {current_device_name}")
else:
    print("\nGPU не доступно.")
