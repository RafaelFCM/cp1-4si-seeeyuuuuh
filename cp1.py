#!/usr/bin/python
# -*- coding: utf-8 -*-

# Programa simples com camera webcam e opencv
import cv2
import numpy as np
import os
import time

def region_of_interest(img, vertices):
    """
    Aplica uma máscara para manter apenas a região de interesse definida pelos vértices
    """
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [vertices], 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

def calcular_inclinacao(linha):
    """
    Calcula a inclinação de uma linha
    Retorna a inclinação em graus
    """
    x1, y1, x2, y2 = linha[0]
    if x2 - x1 == 0:
        return 90.0
    return np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi

def detectar_curva(left_lines, right_lines, width):
    """
    Detecta a direção e intensidade da curva baseado nas linhas detectadas
    """
    # Se não houver linhas suficientes, considera como reta
    if len(left_lines) < 2 and len(right_lines) < 2:
        return "Reta", 0
    
    # Calcula inclinações médias
    left_slopes = []
    right_slopes = []
    
    for line in left_lines:
        slope = calcular_inclinacao(line)
        if -85 < slope < 85:  # Filtra linhas muito verticais
            left_slopes.append(slope)
            
    for line in right_lines:
        slope = calcular_inclinacao(line)
        if -85 < slope < 85:  # Filtra linhas muito verticais
            right_slopes.append(slope)
    
    # Se não houver inclinações válidas suficientes
    if not left_slopes and not right_slopes:
        return "Reta", 0
    
    # Calcula médias das inclinações
    left_mean = np.mean(left_slopes) if left_slopes else 0
    right_mean = np.mean(right_slopes) if right_slopes else 0
    
    # Calcula a diferença entre as inclinações médias
    if left_slopes and right_slopes:
        diff = abs(left_mean) - abs(right_mean)
    elif left_slopes:
        diff = left_mean
    elif right_slopes:
        diff = -right_mean
    else:
        return "Reta", 0
    
    # Lógica para determinar a direção da curva
    threshold = 5  # Limiar para considerar uma curva
    
    if abs(diff) < threshold:
        return "Reta", 0
    else:
        intensidade = abs(diff)
        if diff > 0:
            if intensidade < 15:
                return "Curva Suave Esquerda", intensidade
            else:
                return "Curva Acentuada Esquerda", intensidade
        else:
            if intensidade < 15:
                return "Curva Suave Direita", intensidade
            else:
                return "Curva Acentuada Direita", intensidade

# Carrega o modelo YOLO pré-treinado
def carregar_yolo():
    """
    Carrega o modelo YOLO para detecção de veículos
    """
    # Verifica se os arquivos do modelo existem, se não, faz o download
    if not os.path.exists('yolov3.weights'):
        print("Baixando arquivo yolov3.weights...")
        os.system('curl -o yolov3.weights https://pjreddie.com/media/files/yolov3.weights')
    
    if not os.path.exists('yolov3.cfg'):
        print("Baixando arquivo yolov3.cfg...")
        os.system('curl -o yolov3.cfg https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg')
    
    if not os.path.exists('coco.names'):
        print("Baixando arquivo coco.names...")
        os.system('curl -o coco.names https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names')
    
    # Carrega os nomes das classes
    with open('coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    # Carrega a rede neural
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
    
    # Configura os parâmetros da rede
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    return net, output_layers, classes

def detectar_veiculos(img, net, output_layers, classes):
    """
    Detecta veículos na imagem usando YOLO
    """
    height, width = img.shape[:2]
    
    # Diminui o tamanho pra ficar mais rápido
    scale = 0.5
    small_img = cv2.resize(img, (int(width * scale), int(height * scale)))
    
    blob = cv2.dnn.blobFromImage(small_img, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.6 and classes[class_id] in ['car', 'truck', 'bus', 'motorcycle']:
                center_x = int(detection[0] * width * scale)
                center_y = int(detection[1] * height * scale)
                w = int(detection[2] * width * scale)
                h = int(detection[3] * height * scale)
                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([int(x/scale), int(y/scale), int(w/scale), int(h/scale)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            
            relative_distance = 1.0 - (h / height)
            
            # Muda a cor baseado na distância do carro
            if relative_distance < 0.3:
                color = (0, 0, 255)  # vermelho = perigo
                warning = "PERIGO!"
            elif relative_distance < 0.6:
                color = (0, 255, 255)  # amarelo = atenção
                warning = "ATENÇÃO"
            else:
                color = (0, 255, 0)  # verde = safe
                warning = "OK"
            
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f"{label}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return img

def image_da_webcam(img, net, output_layers, classes, frame_count):
    """
    Função principal de processamento de imagem
    """
    height, width = img.shape[:2]
    
    # Região que vai analisar na imagem (tipo um trapézio)
    roi_vertices = np.array([
        [(0, height), 
         (width//2 - 50, height//2 + 50), 
         (width//2 + 50, height//2 + 50), 
         (width, height)]], dtype=np.int32)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    roi = region_of_interest(edges, roi_vertices)
    
    # Detecta as linhas da pista
    lines = cv2.HoughLinesP(
        roi,
        rho=1,
        theta=np.pi/180,
        threshold=30,
        minLineLength=60,
        maxLineGap=30
    )
    
    line_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 5)
    
    # Junta a imagem original com as linhas detectadas
    result = cv2.addWeighted(img, 0.8, line_img, 1.0, 0)
    
    # Detecta os carros a cada 3 frames pra não ficar pesado
    if frame_count % 3 == 0:
        result = detectar_veiculos(result, net, output_layers, classes)
    
    # 1. Obtém as dimensões da imagem
    height, width = result.shape[:2]
    
    # 2. Define a região de interesse (trapézio)
    roi_vertices = np.array([
        [(50, height),
         (width//2 - 45, height//2 + 60),
         (width//2 + 45, height//2 + 60),
         (width - 50, height)]], dtype=np.int32)
    
    # 3. Converte para escala de cinza
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    
    # 4. Aplica blur para reduzir ruído
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 5. Detecta bordas usando Canny
    edges = cv2.Canny(blur, 50, 150)
    
    # 6. Aplica a máscara da região de interesse
    roi = region_of_interest(edges, roi_vertices)
    
    # 7. Detecta linhas usando transformada de Hough
    lines = cv2.HoughLinesP(
        roi,
        rho=1,
        theta=np.pi/180,
        threshold=30,
        minLineLength=60,
        maxLineGap=30
    )
    
    # 8. Cria uma imagem para desenhar as linhas detectadas
    line_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Listas para armazenar as linhas da esquerda e direita
    left_lines = []
    right_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Separa as linhas em esquerda e direita baseado na posição x
            if x1 < width//2 and x2 < width//2:
                left_lines.append(line)
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 5)  # Vermelho para esquerda
            elif x1 > width//2 and x2 > width//2:
                right_lines.append(line)
                cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 5)  # Azul para direita
    
    # 9. Combina a imagem original com as linhas detectadas
    result = cv2.addWeighted(result, 0.8, line_img, 1.0, 0)
    
    # 10. Detecta a curva e sua intensidade
    direcao_curva, intensidade = detectar_curva(left_lines, right_lines, width)
    
    # 11. Adiciona informações visuais
    # Cor do texto baseada na intensidade da curva
    if intensidade > 15:
        cor_texto = (0, 0, 255)  # Vermelho para curvas acentuadas
    else:
        cor_texto = (0, 255, 0)  # Verde para curvas suaves ou reta
    
    # Desenha a região de interesse
    cv2.polylines(result, [roi_vertices], True, (0, 255, 255), 2)
    
    # Adiciona texto com informações da curva
    cv2.putText(result, f"{direcao_curva}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor_texto, 2)
    
    if intensidade > 0:
        cv2.putText(result, f"Intensidade: {intensidade:.1f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor_texto, 2)
    
    # Mostra FPS e velocidade (fonte menor e mais compacto)
    cv2.putText(result, f"FPS: {fps:.1f} | {velocidade}x", 
                (result.shape[1] - 150, 30),  # Mais pro canto
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5,  # Fonte menor
                (0, 255, 0), 
                2)  # Linha mais fina
    
    # Legenda dos controles (menor e abaixo do FPS)
    cv2.putText(result, "Controlar velocidade: + ou - | ESC (sair)", 
                (result.shape[1] - 350, 60),  # Alinhado abaixo do FPS
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5,  # Fonte ainda menor
                (255, 255, 255), 
                1)  # Linha mais fina
    
    return result

# Configuração inicial
cv2.namedWindow("Sistema de Assistencia ao Motorista", cv2.WINDOW_NORMAL)

# Carrega o modelo YOLO
print("Carregando modelo YOLO...")
net, output_layers, classes = carregar_yolo()
print("Modelo YOLO carregado com sucesso!")

# Define a entrada de vídeo
vc = cv2.VideoCapture("project_video.mp4")
# vc = cv2.VideoCapture("video_simulador2.mp4")
#
# Obtém as dimensões originais do vídeo
original_width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calcula as dimensões para exibição mantendo a proporção
display_height = 848 * 2  # 1696
display_width = 480 * 2   # 960

# Configura o tamanho inicial da janela
cv2.resizeWindow("Sistema de Assistencia ao Motorista", display_width, display_height)

# Variáveis para controle de FPS e velocidade
frame_count = 0
start_time = time.time()
fps = 0
# Fator de velocidade: 2 = 2x mais rápido, 3 = 3x mais rápido, etc.
velocidade = 3  # Ajuste este valor para controlar a velocidade

if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False

while rval:
    frame_count += 1
    
    # Calcula FPS
    if frame_count % 30 == 0:
        end_time = time.time()
        fps = 30 / (end_time - start_time)
        start_time = time.time()
    
    # Processa o frame
    processed_img = image_da_webcam(frame, net, output_layers, classes, frame_count)
    
    
    
    # Redimensiona a imagem usando interpolação de alta qualidade
    display_img = cv2.resize(processed_img, 
                           (display_width, display_height),
                           interpolation=cv2.INTER_LANCZOS4)
    
    # Mostra a imagem processada
    cv2.imshow("Sistema de Assistencia ao Motorista", display_img)
    
    # Pula frames baseado na velocidade desejada
    current_frame = vc.get(cv2.CAP_PROP_POS_FRAMES)
    vc.set(cv2.CAP_PROP_POS_FRAMES, current_frame + velocidade - 1)
    
    rval, frame = vc.read()
    key = cv2.waitKey(1)
    
    # Controles de velocidade com teclado
    if key == ord('+') or key == ord('='): # Tecla + aumenta velocidade
        velocidade += 1
    elif key == ord('-') and velocidade > 1: # Tecla - diminui velocidade
        velocidade -= 1
    elif key == 27: # ESC para sair
        break

cv2.destroyAllWindows()
vc.release()