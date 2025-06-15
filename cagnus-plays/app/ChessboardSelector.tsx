"use client";
import React, { useRef, useEffect, useState } from "react";

const VIDEO_URL = "http://localhost:5001/video_feed";

export default function ChessboardSelector() {
  const imgRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [corners, setCorners] = useState<{ x: number; y: number }[]>([]);
  const [dimensions, setDimensions] = useState({ width: 1280, height: 720 }); // Default, will update on image load

  // Draw markers on canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (canvas && img) {
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      corners.forEach((corner, i) => {
        ctx.beginPath();
        ctx.arc(corner.x, corner.y, 16, 0, 2 * Math.PI); // Larger marker
        ctx.fillStyle = "red";
        ctx.fill();
        ctx.font = "32px Arial";
        ctx.fillText(`${i + 1}`, corner.x + 20, corner.y - 20);
      });
    }
  }, [corners, dimensions]);

  // Handle click
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img) return;
    const rect = canvas.getBoundingClientRect();
    const x = ((e.clientX - rect.left) * img.naturalWidth) / canvas.width;
    const y = ((e.clientY - rect.top) * img.naturalHeight) / canvas.height;
    if (corners.length < 4) {
      setCorners([...corners, { x, y }]);
    }
  };

  // Send corners to backend
  const sendCorners = async () => {
    await fetch("http://localhost:5001/set_corners", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ corners }),
    });
    alert("Corners sent!");
  };

  // Reset selection
  const resetCorners = () => setCorners([]);

  // On image load, update dimensions
  const handleImgLoad = () => {
    if (imgRef.current) {
      setDimensions({
        width: imgRef.current.naturalWidth,
        height: imgRef.current.naturalHeight,
      });
      if (canvasRef.current) {
        canvasRef.current.width = imgRef.current.naturalWidth;
        canvasRef.current.height = imgRef.current.naturalHeight;
      }
    }
  };

  return (
    <div style={{ position: "relative", width: "100%", height: "auto" }}>
      <img
        ref={imgRef}
        src={VIDEO_URL}
        alt="Live Chessboard"
        style={{ width: "100%", height: "auto", display: "block" }}
        onLoad={handleImgLoad}
      />
      <canvas
        ref={canvasRef}
        width={dimensions.width}
        height={dimensions.height}
        style={{
          position: "absolute",
          left: 0,
          top: 0,
          width: "100%",
          height: "100%",
          pointerEvents: "auto",
        }}
        onClick={handleCanvasClick}
      />
      <div style={{ marginTop: 8 }}>
        <button onClick={sendCorners} disabled={corners.length !== 4}>
          Send Corners
        </button>
        <button onClick={resetCorners} style={{ marginLeft: 8 }}>
          Reset
        </button>
      </div>
    </div>
  );
}
