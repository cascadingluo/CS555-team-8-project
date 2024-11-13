document.addEventListener("DOMContentLoaded", () => {
    const addReminderBtn = document.getElementById("addReminderBtn");
    const reminderList = document.getElementById("reminderList");

    // Function to create a new reminder item
    function createReminderItem() {
        // Create a div for the reminder
        const reminderItem = document.createElement("div");
        reminderItem.classList.add("reminder-item");

        // Set the HTML content for the reminder
        reminderItem.innerHTML = `
            <div class="reminder-content">
                <h3>Reminder: <input type="text" class="reminder-title" value="click on edit reminder to modify" disabled></h3>
                <div>
                    <span class="reminder-label">Time:</span>
                    <input type="time" class="reminder-time" value="19:00" disabled>
                </div>
                <div>
                    <span class="reminder-label">Date:</span>
                    <select class="reminder-date" disabled>
                        <option>Monday</option>
                        <option>Tuesday</option>
                        <option>Wednesday</option>
                        <option>Thursday</option>
                        <option>Friday</option>
                        <option>Saturday</option>
                        <option>Sunday</option>
                    </select>
                </div>
                <div>
                    <span class="reminder-label">Recurrence:</span>
                    <select class="reminder-frequency" disabled>
                        <option>Just Once</option>
                        <option>Reoccurring</option>
                    </select>
                </div>
            </div>
            <button class="edit-btn">Edit Reminder</button>
            <button class="save-btn hidden">Save Reminder</button>
            <button class="remove-btn">Remove Reminder</button>
        `;

        // Add event listeners for Edit, Save, and Remove buttons
        const editBtn = reminderItem.querySelector(".edit-btn");
        const saveBtn = reminderItem.querySelector(".save-btn");
        const removeBtn = reminderItem.querySelector(".remove-btn");

        editBtn.addEventListener("click", () => toggleEditMode(reminderItem, true));
        saveBtn.addEventListener("click", () => toggleEditMode(reminderItem, false));
        removeBtn.addEventListener("click", () => reminderItem.remove());

        // Add the reminder item to the reminder list
        reminderList.insertBefore(reminderItem, addReminderBtn);
    }

    // Function to toggle edit mode on a reminder item
    function toggleEditMode(reminderItem, isEditing) {
        const title = reminderItem.querySelector(".reminder-title");
        const time = reminderItem.querySelector(".reminder-time");
        const date = reminderItem.querySelector(".reminder-date");
        const frequency = reminderItem.querySelector(".reminder-frequency");
        const editBtn = reminderItem.querySelector(".edit-btn");
        const saveBtn = reminderItem.querySelector(".save-btn");

        // Toggle between edit and save mode
        title.disabled = !isEditing;
        time.disabled = !isEditing;
        date.disabled = !isEditing;
        frequency.disabled = !isEditing;
        editBtn.classList.toggle("hidden", isEditing);
        saveBtn.classList.toggle("hidden", !isEditing);

        if (!isEditing) {
            // Save the edited values (optional: add more functionality here if needed)
            console.log("Saved Reminder:", {
                title: title.value,
                time: time.value,
                date: date.value,
                frequency: frequency.value
            });
        }
    }

    // Add event listener to the "Add Reminder" button
    addReminderBtn.addEventListener("click", createReminderItem);
});
