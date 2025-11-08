import express from "express";
import { recordListenHistory, getUserListeningHistory } from "../controller/history.controller.js";

const router = express.Router();
router.post("/", recordListenHistory);
router.get("/:userId", getUserListeningHistory);

export default router;
