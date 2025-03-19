import cv2
import numpy as np

cap = cv2.VideoCapture("q1A.mp4")

def detectar_formas(frame):
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
  
    lower_color = np.array([0, 50, 50])  
    upper_color = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, mask

def calcular_centro_de_massa(contorno):
    
    M = cv2.moments(contorno)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return (cx, cy)
    return None

def detectar_colisao(contornos):
   
    for i, cnt1 in enumerate(contornos):
        for j, cnt2 in enumerate(contornos):
            if i != j:  
                # Verifica se os retângulos delimitadores das formas se sobrepõem
                if cv2.boundingRect(cnt1) == cv2.boundingRect(cnt2):
                    return True
    return False

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    contornos, mask = detectar_formas(frame)
    
    for cnt in contornos:
       
        cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 3)
        
        
        centro = calcular_centro_de_massa(cnt)
        if centro:
            cx, cy = centro
            size = 20
            color = (128, 128, 0)
            
            
            cv2.line(frame, (cx - size, cy), (cx + size, cy), color, 5)
            cv2.line(frame, (cx, cy - size), (cx, cy + size), color, 5)
           
            cv2.putText(frame, f"Centro: ({cx},{cy})", (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 50, 0), 2)

   
    if detectar_colisao(contornos):
        cv2.putText(frame, "COLISÃO DETECTADA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Exibindo a imagem com os resultados
    cv2.imshow("Feed", frame)
    
    # Espera pela tecla 'ESC' para encerrar
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

# Libera o vídeo e fecha as janelas
cap.release()
cv2.destroyAllWindows()
