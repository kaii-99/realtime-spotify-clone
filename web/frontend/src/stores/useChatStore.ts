import { axiosInstance } from "@/lib/axios";
import { Message, User } from "@/types";
import { create } from "zustand";
import { io } from "socket.io-client";

interface GroupPlaylistSong {
	id: string;
	title: string;
	artist?: string;
	imageUrl?: string;
	audioUrl?: string;
}

interface GroupPlaylist {
	id: string;
	name: string;
	songs: GroupPlaylistSong[];
}

interface ChatStore {
	users: User[];
	isLoading: boolean;
	error: string | null;
	socket: any;
	isConnected: boolean;
	onlineUsers: Set<string>;
	userActivities: Map<string, string>;
	messages: Message[];
	selectedUser: User | null;
	groupPlaylists: any[];
	groupPlaylistSongs: GroupPlaylist | null;

	fetchUsers: () => Promise<void>;
	initSocket: (userId: string) => void;
	disconnectSocket: () => void;
	sendMessage: (receiverId: string, senderId: string, content: string) => void;
	fetchMessages: (userId: string) => Promise<void>;
	setSelectedUser: (user: User | null) => void;

	createGroupPlaylist: (data: { name: string; members: string[]; user: string }) => Promise<void>;
	fetchGroupPlaylist: (data: { user: string }) => Promise<void>;
	FetchSongGroupPlaylist: (data: { groupId: string }) => Promise<void>;
	AddSongGroupPlaylist: (data: { groupId: string, songId: string }) => Promise<void>;
}

const baseURL = import.meta.env.MODE === "development" ? "http://localhost:5000" : "/";

const socket = io(baseURL, {
	autoConnect: false, // only connect if user is authenticated
	withCredentials: true,
});

export const useChatStore = create<ChatStore>((set, get) => ({
	users: [],
	isLoading: false,
	error: null,
	socket: socket,
	isConnected: false,
	onlineUsers: new Set(),
	userActivities: new Map(),
	messages: [],
	selectedUser: null,
	groupPlaylists: [],
	groupPlaylistSongs: null,

	setSelectedUser: (user) => set({ selectedUser: user }),

	fetchUsers: async () => {
		set({ isLoading: true, error: null });
		try {
			const response = await axiosInstance.get("/users");
			set({ users: response.data });
		} catch (error: any) {
			set({ error: error.response.data.message });
		} finally {
			set({ isLoading: false });
		}
	},

	initSocket: (userId) => {
		if (!get().isConnected) {
			socket.auth = { userId };
			socket.connect();

			socket.emit("user_connected", userId);

			socket.on("users_online", (users: string[]) => {
				set({ onlineUsers: new Set(users) });
			});

			socket.on("activities", (activities: [string, string][]) => {
				set({ userActivities: new Map(activities) });
			});

			socket.on("user_connected", (userId: string) => {
				set((state) => ({
					onlineUsers: new Set([...state.onlineUsers, userId]),
				}));
			});

			socket.on("user_disconnected", (userId: string) => {
				set((state) => {
					const newOnlineUsers = new Set(state.onlineUsers);
					newOnlineUsers.delete(userId);
					return { onlineUsers: newOnlineUsers };
				});
			});

			socket.on("receive_message", (message: Message) => {
				set((state) => ({
					messages: [...state.messages, message],
				}));
			});

			socket.on("message_sent", (message: Message) => {
				set((state) => ({
					messages: [...state.messages, message],
				}));
			});

			socket.on("activity_updated", ({ userId, activity }) => {
				set((state) => {
					const newActivities = new Map(state.userActivities);
					newActivities.set(userId, activity);
					return { userActivities: newActivities };
				});
			});

			set({ isConnected: true });
		}
	},

	disconnectSocket: () => {
		if (get().isConnected) {
			socket.disconnect();
			set({ isConnected: false });
		}
	},

	sendMessage: async (receiverId, senderId, content) => {
		const socket = get().socket;
		if (!socket) return;

		socket.emit("send_message", { receiverId, senderId, content });
	},

	fetchMessages: async (userId: string) => {
		set({ isLoading: true, error: null });
		try {
			const response = await axiosInstance.get(`/users/messages/${userId}`);
			set({ messages: response.data });
		} catch (error: any) {
			set({ error: error.response.data.message });
		} finally {
			set({ isLoading: false });
		}
	},

	createGroupPlaylist: async ({ name, members, user }) => {
		const clerkId = user
		set({ isLoading: true, error: null });

		try {
			const userRes = await axiosInstance.get(`/users/clerk/${clerkId}`);
			const userId = userRes.data._id;
			const updatedMembers = [...members, userId];

			await axiosInstance.post("/group-playlists", { 
				name, 
				members: updatedMembers,
			});

		} catch (error: any) {
			set({
				error: error?.response?.data?.message || "Failed to create group playlist",
			});
			throw error;

		} finally {
			set({ isLoading: false });
		}
	},

	fetchGroupPlaylist: async ({ user }) => {
		const clerkId = user
		console.log("Backend User:", clerkId)
		set({ isLoading: true, error: null });

		try {
			const userRes = await axiosInstance.get(`/users/clerk/${clerkId}`);
			const userId = userRes.data._id;

			const res = await axiosInstance.get(`/group-playlists/${userId}`);
			set({ groupPlaylists: res.data });

		} catch (error: any) {
			set({
				error: error?.response?.data?.message || "Failed to create group playlist",
			});
			throw error;

		} finally {
			set({ isLoading: false });
		}
	},

	FetchSongGroupPlaylist: async ({ groupId }) => {
		set({ isLoading: true, error: null });

		try {
			const res = await axiosInstance.get(`/group-playlists/fetchsong/${groupId}`);
			set({ groupPlaylistSongs: res.data });

		} catch (error: any) {
			set({
				error: error?.response?.data?.message || "Failed to fetch group playlist songs",
			});
			throw error;

		} finally {
			set({ isLoading: false });
		}
	},

	AddSongGroupPlaylist: async ({ groupId, songId }) => {
		set({ isLoading: true, error: null });

		try {
			await axiosInstance.patch(`/group-playlists/${groupId}/add-song`, { 
				songId: songId,
			});

		} catch (error: any) {
			set({
				error: error?.response?.data?.message || "Failed to add song to group playlist",
			});
			throw error;

		} finally {
			set({ isLoading: false });
		}
	}
}));
