module.exports = {
  content: ["./**/*.{html,js}"],
  theme: {
    extend: {
      // You can add custom scrollbar styles here
      scrollbar: ['rounded']
    },
  },
  plugins: [
    // Include the plugin for scrollbar customization
    require('tailwind-scrollbar'),
  ],
}
