import express from "express";
import { exportListeningData, exportAllSongData } from "../controller/export.controller.js";

const router = express.Router();

router.get("/listenhistory", exportListeningData);
router.get("/allsong", exportAllSongData);

export default router;
