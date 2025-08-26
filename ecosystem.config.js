module.exports = {
  apps: [
    {
      name: "ai-callcenter-api",
      script: "main.py",                     // ชื่อไฟล์ Python ของคุณ
      interpreter: "C:\\Users\\pc.it07\\AppData\\Local\\Programs\\Python\\Python312\\python.exe",  // Python interpreter ของคุณ
      args: "",
      watch: false,
      autorestart: true,
      max_memory_restart: "1G",
      out_file : './logs/out.log',
      error_file: "./logs/error.log",
      log_date_format: "YYYY-MM-DD HH:mm Z",
      env: {
        "PORT": 8001,
        "HOST": "0.0.0.0"
      }
    }
  ]
};