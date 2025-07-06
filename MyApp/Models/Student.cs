    using System;
using System.ComponentModel.DataAnnotations;

    namespace MyApp.Models
    {
        public class Student
        {
            [Key]
            public int Id { get; set; }
            [Required(ErrorMessage = "Tên không được để trống.")]
            public string? Name { get; set; }
            [Required(ErrorMessage = "Thời gian check-in không được để trống.")]
            public string? CheckinTime { get; set; }
        }
    } 