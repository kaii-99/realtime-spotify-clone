import express from "express";
import { recordListenHistory } from "../controller/history.controller.js";

const router = express.Router();
router.post("/", recordListenHistory);

export default router;
